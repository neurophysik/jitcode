#!/usr/bin/python3
# -*- coding: utf-8 -*-

from scipy.integrate import ode
from scipy.integrate._ode import find_integrator
from warnings import warn
from traceback import format_exc
from types import FunctionType, BuiltinFunctionType
from inspect import signature
from itertools import count

from numpy import hstack, log
import numpy as np
import symengine

from jitcxde_common import jitcxde
from jitcxde_common.modules import module_from_path
from jitcxde_common.helpers import sympify_helpers, sort_helpers
from jitcxde_common.numerical import random_direction, orthonormalise
from jitcxde_common.symbolic import collect_arguments, ordered_subs, replace_function
from jitcxde_common.transversal import GroupHandler

#: the symbol for the state that must be used to define the differential equation. It is a function and the integer argument denotes the component. You may just as well define an analogous function directly with SymEngine or SymPy, but using this function is the best way to get the most of future versions of JiTCODE, in particular avoiding incompatibilities.
y = symengine.Function("y")

#: the symbol for time for defining the differential equation. If your differential equation has no explicit time dependency (“autonomous system”), you do not need this. You may just as well define an analogous symbol directly with SymEngine or SymPy, but using this function is the best way to get the most of future versions of JiTCODE, in particular avoiding incompatibilities.
t = symengine.Symbol("t", real=True)

class UnsuccessfulIntegration(Exception):
	"""
		This exception is raised when the integrator cannot meet the accuracy and step-size requirements. If you want to know the exact state of your system before the integration fails or similar, catch this exception.
	"""
	pass

def _can_use_jacobian(integratorname):
	integrator = find_integrator(integratorname)
	parameters = signature(integrator.__init__).parameters
	return "with_jacobian" in parameters

def _is_C(function):
	return isinstance(function, BuiltinFunctionType)

def _is_lambda(function):
	return isinstance(function, FunctionType)

def _jac_from_f_with_helpers(f, helpers, simplify, n):
	dependent_helpers = [[] for i in range(n)]
	for i in range(n):
		for helper in helpers:
			derivative = helper[1].diff(y(i))
			for other_helper in dependent_helpers[i]:
				derivative += helper[1].diff(other_helper[0]) * other_helper[1]
			if derivative:
				dependent_helpers[i].append( (helper[0], derivative) )
	
	def line(f_entry):
		for j in range(n):
			entry = f_entry.diff(y(j))
			for helper in dependent_helpers[j]:
				entry += f_entry.diff(helper[0]) * helper[1]
			if simplify:
				entry = entry.simplify(ratio=1.0)
			yield entry
	
	for f_entry in f():
		yield line(f_entry)

def _lambda_wrapper(core_f,control_pars):
	# I would like to do this:
	# return lambda t,Y,*c_pars: core_f(np.hstack([t,Y,c_pars]))
	# but since SciPy’s ODE fails to handle variadic functions, it has to be this:
	
	n = len(control_pars)
	if   n==0: return lambda t,Y          : core_f(hstack([t,Y          ]))
	elif n==1: return lambda t,Y,p        : core_f(hstack([t,Y,p        ]))
	elif n==2: return lambda t,Y,p,q      : core_f(hstack([t,Y,p,q      ]))
	elif n==3: return lambda t,Y,p,q,r    : core_f(hstack([t,Y,p,q,r    ]))
	elif n==4: return lambda t,Y,p,q,r,s  : core_f(hstack([t,Y,p,q,r,s  ]))
	elif n==5: return lambda t,Y,p,q,r,s,u: core_f(hstack([t,Y,p,q,r,s,u]))
	else:
		cpars = ",".join( par.name for par in control_pars )
		return eval(
				"lambda t,Y,%s: core_f(hstack([t,Y,%s]))" % (cpars,cpars),
				{"core_f":core_f, "hstack":hstack}
			)

class jitcode(ode,jitcxde):
	"""
	Parameters
	----------
	f_sym : iterable of symbolic expressions or generator function yielding symbolic expressions
		The `i`-th element is the `i`-th component of the value of the ODE’s derivative :math:`f(t,y)`.
	
	helpers : list of length-two iterables, each containing a symbol and a symbolic expression
		Each helper is a variable that will be calculated before evaluating the derivative and can be used in the latter’s computation. The first component of the tuple is the helper’s symbol as referenced in the derivative or other helpers, the second component describes how to compute it from `t`, `y` and other helpers. This is for example useful to realise a mean-field coupling, where the helper could look like `(mean, sum(y(i) for i in range(100))/100)`. (See `example_2` for an example.)
	
	wants_jacobian : boolean
		Tell JiTCODE to calculate and compile the Jacobian. For vanilla use, you do not need to bother about this as this is automatically set to `True` if the selected method of integration desires the Jacobian. However, it is sometimes useful if you want to manually apply some code-generation steps (e.g., to apply some tweaks).
		
	n : integer
		Length of `f_sym`. While JiTCODE can easily determine this itself (and will, if necessary), this may take some time if `f_sym` is a generator function and `n` is large. Take care that this value is correct – if it isn’t, you will not get a helpful error message.
	
	control_pars : iterable of symbols
		Each symbol corresponds to a control parameter that can be used when defining the equations and set after compilation `scipy.ode`’s `set_f_params` or `set_jac_params` (in the same order as given here). Using this makes sense if you need to do a parameter scan with short integrations for each parameter and you are spending a considerable amount of time compiling.
	
	verbose : boolean
		Whether JiTCODE shall give progress reports on the processing steps.
	
	module_location : string
		location of a module file from which functions are to be loaded (see `save_compiled`). If you use this, you need not give `f_sym` as an argument, but in this case you must give `n`. Depending on the arguments you provide, functionalities such as recompiling may not be available; but then the entire point of this option is to avoid these.
	"""
	
	def __init__(self,
				f_sym = (),
				helpers = None,
				wants_jacobian = False,
				n = None,
				control_pars = (),
				verbose = True,
				module_location = None,
			):
		ode.__init__(self,None)
		jitcxde.__init__(self,n,verbose,module_location)
		
		self.f_sym = self._handle_input(f_sym)
		self._f_C_source = False
		self.helpers = sort_helpers(sympify_helpers(helpers or []))
		self._wants_jacobian = wants_jacobian
		self.jac_sym = None
		self._jac_C_source = False
		self._helper_C_source = False
		self.control_pars = control_pars
		self._number_of_jac_helpers = None
		self._number_of_f_helpers = None
		self._number_of_general_helpers = len(self.helpers)
		self._lambda_subs = None
		self._lambda_args = None
		self.general_subs = {
				control_par: symengine.Symbol("parameter_"+control_par.name)
				for control_par in self.control_pars
			}
	
	def check(self, fail_fast=True):
		"""
		Checks for the following mistakes:
		
		* negative arguments of `y`
		* arguments of `y` that are higher than the system’s dimension `n`
		* unused variables
		
		For large systems, this may take some time (which is why it is not run by default).
		
		Parameters
		----------
		fail_fast : boolean
			whether to abort on the first failure. If false, an error is raised only after all problems are printed.
		"""
		
		failed = False
		
		def problem(message):
			failed = True
			if fail_fast:
				raise ValueError(message)
			else:
				print(message)
		
		valid_symbols = [t] + [helper[0] for helper in self.helpers] + list(self.control_pars)
		
		assert self.f_sym(), "f_sym is empty."
		
		for i,entry in enumerate(self.f_sym()):
			for argument in collect_arguments(entry,y):
				if argument[0] < 0:
					problem("y is called with a negative argument (%i) in equation %i." % (argument[0], i))
				if argument[0] >= self.n:
					problem("y is called with an argument (%i) higher than the system’s dimension (%i) in equation %i."  % (argument[0], self.n, i))
			
			for symbol in entry.atoms(symengine.Symbol):
				if symbol not in valid_symbols:
					problem("Invalid symbol (%s) in equation %i."  % (symbol.name, i))
		
		if failed:
			raise ValueError("Check failed.")
	
	def _generate_jac_sym(self):
		if self.jac_sym is None:
			self.generate_jac_sym()
			#self.report("generated symbolic Jacobian")
	
	def generate_jac_sym(self, simplify=True):
		"""
		generates the Jacobian using SymEngine’s differentiation.
		
		Parameters
		----------
		simplify : boolean
			Whether the resulting Jacobian should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`). This is almost always a good thing.
		"""
		
		self.jac_sym = _jac_from_f_with_helpers(self.f_sym, self.helpers, simplify, self.n)
	
	def _default_arguments(self):
		basics = [
			("t", "double const"),
			("Y", "PyArrayObject *__restrict const")
			]
		pars = [("parameter_"+par.name, "double const") for par in self.control_pars]
		return basics + pars
	
	def _generate_f_C(self):
		if not self._f_C_source:
			self.generate_f_C()
			self.report("generated C code for f")
	
	def generate_f_C(self, simplify=None, do_cse=False, chunk_size=100):
		"""
		translates the derivative to C code using SymEngine’s `C-code printer <https://github.com/symengine/symengine/pull/1054>`_.
		
		Parameters
		----------
		simplify : boolean or None
			Whether the derivative should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`) before translating to C code. The main reason why you could want to enable this is if you expect your derivative not to be optimised and not be so large that simplifying takes a considerable amount of time. If `None`, this will be automatically disabled for `n>10`.
		
		do_cse : boolean
			Whether SymPy’s `common-subexpression detection <http://docs.sympy.org/dev/modules/rewriting.html#module-sympy.simplify.cse_main>`_ should be applied before translating to C code. It is almost always better to let the compiler do this (unless you want to set the compiler optimisation to `-O2` or lower): For simple differential equations this should not make any difference to the compiler’s optimisations. For large ones, it may make a difference but also take long. As this requires all entries of `f` at once, it may void advantages gained from using generator functions as an input. Also, this feature uses SymPy and not SymEngine.
		
		chunk_size : integer
			If the number of instructions in the final C code exceeds this number, it will be split into chunks of this size. See `Handling very large differential equations <http://jitcde-common.readthedocs.io/#handling-very-large-differential-equations>`_ on why this is useful and how to best choose this value.
			If smaller than 1, no chunking will happen.
		"""
		
		self._generate_helpers_C()
		
		f_sym_wc = (entry.subs(self.general_subs) for entry in self.f_sym())
		
		if simplify is None:
			simplify = self.n<=10
		if simplify:
			f_sym_wc = (entry.simplify(ratio=1) for entry in f_sym_wc)
		
		arguments = self._default_arguments()
		
		if self._number_of_general_helpers:
			arguments.append(("general_helper","double const *__restrict const"))
		
		if do_cse:
			import sympy
			get_helper = sympy.Function("get_f_helper")
			set_helper = symengine.Function("set_f_helper")
			
			_cse = sympy.cse(
					sympy.Matrix(sympy.sympify(list(f_sym_wc))),
					symbols = (get_helper(i) for i in count())
				)
			more_helpers = symengine.sympify(_cse[0])
			f_sym_wc = symengine.sympify(_cse[1][0])
			
			if more_helpers:
				arguments.append(("f_helper","double *__restrict const"))
				self.render_and_write_code(
					(set_helper(i,helper[1]) for i,helper in enumerate(more_helpers)),
					name = "f_helpers",
					chunk_size = chunk_size,
					arguments = arguments
					)
				self._number_of_f_helpers = len(more_helpers)
		
		set_dy = symengine.Function("set_dy")
		self.render_and_write_code(
			(set_dy(i,entry) for i,entry in enumerate(f_sym_wc)),
			name = "f",
			chunk_size = chunk_size,
			arguments = arguments+[("dY", "PyArrayObject *__restrict const")]
			)
		
		self._f_C_source = True
	
	def _generate_jac_C(self):
		if self._wants_jacobian and not self._jac_C_source:
			self.generate_jac_C()
			self.report("generated C code for Jacobian")
	
	def generate_jac_C(self, do_cse=False, chunk_size=100, sparse=True):
		"""
		translates the symbolic Jacobian to C code using SymEngine’s `C-code printer <https://github.com/symengine/symengine/pull/1054>`_. If the symbolic Jacobian has not been generated, it generates it by calling `generate_jac_sym`.
		
		Parameters
		----------
		
		do_cse : boolean
			Whether SymPy’s `common-subexpression detection <http://docs.sympy.org/dev/modules/rewriting.html#module-sympy.simplify.cse_main>`_ should be applied before translating to C code. It is almost always better to let the compiler do this (unless you want to set the compiler optimisation to `-O2` or lower): For simple differential equations this should not make any difference to the compiler’s optimisations. For large ones, it may make a difference but also take long. As this requires the entire Jacobian at once, it may void advantages gained from using generator functions as an input. Also, this feature uses SymPy and not SymEngine.
		
		chunk_size : integer
			If the number of instructions in the final C code exceeds this number, it will be split into chunks of this size. See `Handling very large differential equations <http://jitcde-common.readthedocs.io/#handling-very-large-differential-equations>`_ on why this is useful and how to best choose this value.
			If smaller than 1, no chunking will happen.
		
		sparse : boolean
			Whether a sparse Jacobian should be assumed for optimisation. Note that this does not mean that the Jacobian is stored, parsed or handled as a sparse matrix. This kind of optimisation would require SciPy’s ODE to be able to handle sparse matrices without structure in the sparseness.
		"""
		
		self._generate_helpers_C()
		self._generate_jac_sym()
		
		jac_sym_wc = symengine.Matrix([ [entry.subs(self.general_subs) for entry in line] for line in self.jac_sym ])
		self.sparse_jac = sparse
		
		arguments = self._default_arguments()
		if self._number_of_general_helpers:
			arguments.append(("general_helper","double const *__restrict const"))
		
		if do_cse:
			import sympy
			get_helper = sympy.Function("get_jac_helper")
			set_helper = symengine.Function("set_jac_helper")
			
			_cse = sympy.cse(
					sympy.sympify(jac_sym_wc),
					symbols = (get_helper(i) for i in count())
				)
			more_helpers = symengine.sympify(_cse[0])
			jac_sym_wc = symengine.sympify(_cse[1][0])
			
			if more_helpers:
				arguments.append(("jac_helper","double *__restrict const"))
				self.render_and_write_code(
					(set_helper(i, helper[1]) for i,helper in enumerate(more_helpers)),
					name = "jac_helpers",
					chunk_size = chunk_size,
					arguments = arguments
					)
				self._number_of_jac_helpers = len(more_helpers)
		
		jac_sym_wc = map(symengine.sympify,jac_sym_wc.tolist())
		
		set_dfdy = symengine.Function("set_dfdy")
		
		self.render_and_write_code(
			(
				set_dfdy(i,j,entry)
				for i,line in enumerate(jac_sym_wc)
				for j,entry in enumerate(line)
				if ( (entry != 0) or not self.sparse_jac )
			),
			name = "jac",
			chunk_size = chunk_size,
			arguments = arguments+[("dfdY", "PyArrayObject *__restrict const")]
		)
		
		self._jac_C_source = True
	
	def _generate_helpers_C(self):
		if self.helpers and not self._helper_C_source:
			self.generate_helpers_C()
			self.report("generated C code for helpers")
	
	def generate_helpers_C(self, chunk_size=100):
		"""
		translates the helpers to C code using SymEngine’s `C-code printer <https://github.com/symengine/symengine/pull/1054>`_.
		
		Parameters
		----------
		chunk_size : integer
			If the number of instructions in the final C code exceeds this number, it will be split into chunks of this size. See `large_systems` on why this is useful.
			
			If there is an obvious grouping of your helpers, the group size suggests itself for `chunk_size`.
			
			If smaller than 1, no chunking will happen.
		"""
		
		if self.helpers:
			get_helper = symengine.Function("get_general_helper")
			set_helper = symengine.Function("set_general_helper")
			
			for i,helper in enumerate(self.helpers):
				self.general_subs[helper[0]] = get_helper(i)
			self.render_and_write_code(
					(set_helper(i, helper[1].subs(self.general_subs)) for i,helper in enumerate(self.helpers)),
					name = "general_helpers",
					chunk_size = chunk_size,
					arguments = self._default_arguments() + [("general_helper","double *__restrict const")]
				)
		
		self._helper_C_source = True
	
	def _compile_C(self):
		if (not _is_C(self.f)) or self._lacks_jacobian:
			self.compile_C()
			self.report("compiled C code")
	
	def compile_C(
		self,
		extra_compile_args = None,
		extra_link_args = None,
		verbose = False,
		modulename = None
		):
		"""
		compiles the C code (using `Setuptools <http://pythonhosted.org/setuptools/>`_) and loads the compiled functions. If no C code exists, it is generated by calling `generate_f_C` and `generate_jac_C`.
		For detailed information on the arguments and other ways to tweak the compilation, read `these notes <jitcde-common.readthedocs.io>`_.
		
		Parameters
		----------
		extra_compile_args : iterable of strings
		extra_link_args : iterable of strings
			Arguments to be handed to the C compiler or linker, respectively.
		verbose : boolean
			Whether the compiler commands shall be shown. This is the same as Setuptools’ `verbose` setting.
		modulename : string or `None`
			The name used for the compiled module.
		"""
		
		self._generate_helpers_C()
		self._generate_f_C()
		self._generate_jac_C()
		
		self._process_modulename(modulename)
		
		self._render_template(
			n = self.n,
			has_Jacobian = self._jac_C_source,
			number_of_f_helpers = self._number_of_f_helpers or 0,
			number_of_jac_helpers = self._number_of_jac_helpers or 0,
			number_of_general_helpers = len(self.helpers),
			sparse_jac = self.sparse_jac if self._jac_C_source else None,
			control_pars = [par.name for par in self.control_pars]
			)
		
		self._compile_and_load(verbose,extra_compile_args,extra_link_args)
	
	def _generate_f_lambda(self):
		if not _is_lambda(self.f):
			self.generate_f_lambda()
			self.report("generated lambdified f")
	
	def _prepare_lambdas(self):
		if self._lambda_subs is None or self._lambda_args is None:
			if self.helpers:
				warn("Lambdification handles helpers by plugging them in. This may be very inefficient")
			
			self._lambda_subs = list(reversed(self.helpers))
			self._lambda_args = [t]
			for i in range(self.n):
				symbol = symengine.Symbol("dummy_argument_%i"%i)
				self._lambda_args.append(symbol)
				self._lambda_subs.append((y(i),symbol))
			self._lambda_args.extend(self.control_pars)
	
	def generate_f_lambda(self, simplify=None, do_cse=False):
		"""
		translates the symbolic derivative to a function using SymEngines `Lambdify` or `LambdifyCSE`.
		
		Parameters
		----------
		simplify : boolean
			Whether the derivative should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`) before translating to C code. The main reason why you could want to enable this is if you expect your derivative not to be optimised and not be so large that simplifying takes a considerable amount of time. If `None`, this will be automatically disabled for `n>10`.
		
		do_cse : boolean
			Whether a common-subexpression detection, namely `LambdifyCSE`, should be used.
		"""
		
		self._prepare_lambdas()
		
		f_sym_wc = (ordered_subs(entry,self._lambda_subs) for entry in self.f_sym())
		
		if simplify is None:
			simplify = self.n<=10
		if simplify:
			f_sym_wc = (entry.simplify(ratio=1.0) for entry in f_sym_wc)
		
		lambdify = symengine.LambdifyCSE if do_cse else symengine.Lambdify
		core_f = lambdify(self._lambda_args,list(f_sym_wc))
		# self.f = lambda t,Y,*c_pars: core_f(np.hstack([t,Y,c_pars]))
		self.f = _lambda_wrapper(core_f,self.control_pars)
	
	def _generate_jac_lambda(self):
		if not _is_lambda(self.jac):
			self.generate_jac_lambda()
			self.report("generated lambdified Jacobian")
	
	def generate_jac_lambda(self,do_cse=False):
		"""
		translates the symbolic Jacobain to a function using SymEngines `Lambdify` or `LambdifyCSE`. If the symbolic Jacobian has not been generated, it is generated by calling `generate_jac_sym`.
		
		Parameters
		----------
		do_cse : boolean
			Whether a common-subexpression detection, namely `LambdifyCSE`, should be used.
		"""
		
		self._prepare_lambdas()
		self._generate_jac_sym()
		
		jac_matrix = symengine.Matrix([
				[ordered_subs(entry,self._lambda_subs) for entry in line]
				for line in self.jac_sym
			])
		
		lambdify = symengine.LambdifyCSE if do_cse else symengine.Lambdify
		core_jac = lambdify(self._lambda_args,jac_matrix)
		# self.jac = lambda t,Y,*c_pars: core_jac(np.hstack([t,Y,c_pars]))
		self.jac = _lambda_wrapper(core_jac,self.control_pars)
	
	def generate_lambdas(self):
		"""
		If they do not already exists, this generates lambdified functions by calling `self.generate_f_lambda()` and, if wanted, `generate_jac_lambda()`.
		"""
		
		self._generate_f_lambda()
		if self._wants_jacobian:
			self._generate_jac_lambda()
		self.compile_attempt = False
	
	
	@property
	def is_initiated(self):
		return (self.f is not None) and not self._lacks_jacobian
	
	@property
	def _lacks_jacobian(self):
		return self._wants_jacobian and (self.jac is None)
	
	def _initiate(self):
		if self.compile_attempt is None:
			self._attempt_compilation(reset=False)
		
		if not self.is_initiated:
			if self.compile_attempt:
				self.f = self.jitced.f
				if hasattr(self.jitced,"jac"):
					self.jac = self.jitced.jac
			else:
				self.generate_lambdas()
		
		if self._lacks_jacobian:
			self.compile_attempt = None
			self._initiate()
	
	def generate_functions(self):
		"""
		The central function-generating function. Tries to compile the derivative and, if wanted, the Jacobian. If this fails, it generates lambdified functions as a fallback.
		"""
		
		self._initiate()
	
	def set_initial_value(self, initial_value, time=0.0):
		"""
		Same as the analogous function in SciPy’s ODE. Note that this calls `set_integrator`, if no integrator has been set yet.
		"""
		
		if self.n != len(initial_value):
			raise ValueError("The dimension of the initial value does not match the dimension of your differential equations.")
		
		super(jitcode, self).set_initial_value(initial_value, time)
		return self
	
	def set_integrator(self,name,nsteps=10**6,**integrator_params):
		"""
		Same as the analogous function in SciPy’s ODE, except that it automatically generates the derivative and Jacobian, if they do not exist yet and are needed. Also note that the parameter `nsteps` is set to a much higher value per default.
		"""
		
		if name == 'zvode':
			raise NotImplementedError("JiTCODE does not natively support complex numbers yet.")
		
		self._wants_jacobian |= _can_use_jacobian(name)
		self._initiate()
		
		super(jitcode, self).set_integrator(name,nsteps=nsteps,**integrator_params)
		return self
	
	def set_f_params(self, *args):
		"""
		Same as for SciPy’s ODE, except that it also sets the parameters of the Jacobian (because they should be the same anyway).
		"""
		super(jitcode, self).set_f_params  (*args)
		super(jitcode, self).set_jac_params(*args)
		return self
	
	def set_jac_params(self, *args):
		"""
		Same as for SciPy’s ODE, except that it also sets the parameters of `f` (because they should be the same anyway).
		"""
		super(jitcode, self).set_f_params  (*args)
		super(jitcode, self).set_jac_params(*args)
		return self
	
	# This wrapper exists only to avoid pointless errors and confusing warnings as well as raising a proper exception in case the integration fails.
	def integrate(self,t,step=False,relax=False):
		if t>self.t or step or relax:
			result = super(jitcode,self).integrate(t,step,relax)
			if self.successful():
				return result
			else:
				raise UnsuccessfulIntegration
		elif t==self.t:
			return self.y
		else:
			raise ValueError("Target time smaller than current time. Cannot integrate backwards in time")

class jitcode_lyap(jitcode):
	"""
	the handling is the same as that for `jitcode` except for:
	
	Parameters
	----------
	n_lyap : integer
		Number of Lyapunov exponents to calculate. If negative or larger than the dimension of the system, all Lyapunov exponents are calculated.
	
	simplify : boolean
		Whether the differential equations for the tangent vector shall be subjected to SymEngine’s `simplify`. Doing so may speed up the time evolution but may slow down the generation of the code (considerably for large differential equations).
	"""
	
	def __init__( self, f_sym=(), n_lyap=-1, simplify=True, **kwargs ):
		self.n_basic = kwargs.pop("n",None)
		
		f_basic = self._handle_input(f_sym,n_basic=True)
		self._n_lyap = n_lyap if (0<=n_lyap<=self.n_basic) else self.n_basic
		
		helpers = sort_helpers(sympify_helpers( kwargs.pop("helpers",[]) ))
		
		def f_lyap():
			yield from f_basic()
			
			for i in range(self._n_lyap):
				for line in _jac_from_f_with_helpers(
						f = f_basic,
						helpers = helpers,
						simplify = False,
						n = self.n_basic
					):
					expression = sum(
							entry * y(k+(i+1)*self.n_basic)
							for k,entry in enumerate(line)
							if entry
						)
					if simplify:
						expression = expression.simplify(ratio=1.0)
					yield expression
		
		super(jitcode_lyap, self).__init__(
				f_lyap,
				helpers = helpers,
				n = self.n_basic*(self._n_lyap+1),
				**kwargs
			)
		
	def set_initial_value(self, y, t=0.0):
		new_y = [y]
		for _ in range(self._n_lyap):
			new_y.append(random_direction(self.n_basic))
		
		super(jitcode_lyap, self).set_initial_value(hstack(new_y), t)
	
	def norms(self):
		n = self.n_basic
		tangent_vectors = [ self._y[(i+1)*n:(i+2)*n] for i in range(self._n_lyap) ]
		norms = orthonormalise(tangent_vectors)
		if not np.all(np.isfinite(norms)):
			warn("Norms of perturbation vectors for Lyapunov exponents out of numerical bounds. You probably waited too long before renormalising and should call integrate with smaller intervals between steps (as renormalisations happen once with every call of integrate).")
		return norms, tangent_vectors
	
	def integrate(self, *args, **kwargs):
		"""
		Like SciPy’s ODE’s `integrate`, except for orthonormalising the tangent vectors and:
		
		Returns
		-------
		y : one-dimensional NumPy array
			The state of the system. Same as the output of `jitcode`’s `integrate` and `ode`’s `integrate`.
		
		lyaps : one-dimensional NumPy array
			The “local” Lyapunov exponents as estimated from the growth or shrinking of the tangent vectors during the integration time of this very `integrate` command, i.e., :math:`\\frac{\\ln (α_i^{(p)})}{s_i}` in the notation of [BGGS80]_
		
		lyap_vectors : list of one-dimensional NumPy arrays
			The Lyapunov vectors (normalised) after integration.
		"""
		
		old_t = self.t
		super(jitcode_lyap, self).integrate(*args, **kwargs)
		delta_t = self.t-old_t
		norms, tangent_vectors = self.norms()
		lyaps = log(norms) / delta_t
		super(jitcode_lyap, self).set_initial_value(self._y, self.t)
		
		return self._y[:self.n_basic], lyaps, tangent_vectors


class jitcode_transversal_lyap(jitcode):
	"""
	Calculates the largest Lyapunov exponent in orthogonal direction to a predefined synchronisation manifold, i.e. the projection of the tangent vector onto that manifold vanishes. In contrast to `jitcode_restricted_lyap`, this performs some transformations tailored to this specific application that may strongly reduce the number of differential equations and ensure a dynamics on the synchronisation manifold.
	
	The handling is the same as that for `jitcode` except for:
	
	Parameters
	----------
	groups : iterable of iterables of integers
		each group is an iterable of indices that identify dynamical variables that are synchronised on the synchronisation manifold.
	
	simplify : boolean
		Whether the transformed differential equations shall be subjected to SymEngine’s `simplify`. Doing so may speed up the time evolution but may slow down the generation of the code (considerably for large differential equations).
	"""
	
	def __init__( self, f_sym=(), groups=(), simplify=True, **kwargs ):
		self.G = GroupHandler(groups)
		self.n = kwargs.pop("n",None)
		
		f_basic,extracted = self.G.extract_main(self._handle_input(f_sym))
		helpers = sort_helpers(sympify_helpers( kwargs.pop("helpers",[]) ))
		
		z = symengine.Function("z")
		z_vector = [z(i) for i in range(self.n)]
		tangent_vector = self.G.back_transform(z_vector)
		
		def tangent_vector_f():
			for line in _jac_from_f_with_helpers(
					f = f_basic,
					helpers = helpers,
					simplify = False,
					n = self.n
				):
				yield sum(
						entry * tangent_vector[k]
						for k,entry in enumerate(line)
						if entry
					)
		
		substitutions = {
				y(i): z(self.G.map_to_main(i))
				for i in range(self.n)
			}
		
		def finalise(entry):
			entry = entry.subs(substitutions)
			if simplify:
				entry = entry.simplify(ratio=1)
			return replace_function(entry,z,y)
		
		def f_lyap():
			for entry in self.G.iterate(tangent_vector_f()):
				if type(entry)==int:
					yield finalise(extracted[self.G.main_indices[entry]])
				else:
					yield finalise(entry[0]-entry[1])
		
		helpers = ((helper[0],finalise(helper[1])) for helper in helpers)
		
		super(jitcode_transversal_lyap, self).__init__(
				f_lyap,
				helpers = helpers,
				n = self.n,
				**kwargs
			)
		
	def set_initial_value(self, y, t=0.0):
		"""
		Like the analogous function of `jitcode`/`scipy.integrate.ode`, except that only one initial value per group of synchronised components has to be provided (in the same order as the `groups` argument of the constructor).
		"""
		assert len(y)==len(self.G.groups), "Initial state too long. Provide only one value per synchronisation group"
		
		new_y = np.empty(self.n)
		new_y[self.G.main_indices] = y
		new_y[self.G.tangent_indices] = random_direction(len(self.G.tangent_indices))
		super(jitcode_transversal_lyap, self).set_initial_value(new_y, t)
	
	def norm(self):
		tangent_vector = self._y[self.G.tangent_indices]
		norm = np.linalg.norm(tangent_vector)
		tangent_vector /= norm
		if not np.isfinite(norm):
			warn("Norm of perturbation vector for Lyapunov exponent out of numerical bounds. You probably waited too long before renormalising and should call integrate with smaller intervals between steps (as renormalisations happen once with every call of integrate).")
		self._y[self.G.tangent_indices] = tangent_vector
		return norm
	
	def integrate(self, *args, **kwargs):
		"""
		Like SciPy’s ODE’s `integrate`, except for orthonormalising the tangent vector and:
		
		Returns
		-------
		y : one-dimensional NumPy array
			The state of the system. Only one initial value per group of synchronised components is returned (in the same order as the `groups` argument of the constructor).
		
		lyaps : one-dimensional NumPy array
			The “local” Lyapunov exponent as estimated from the growth or shrinking of the tangent vector during the integration time of this very `integrate` command, i.e., :math:`\\frac{\\ln (α_i^{(p)})}{s_i}` in the notation of [BGGS80]_
		"""
		
		old_t = self.t
		super(jitcode_transversal_lyap, self).integrate(*args, **kwargs)
		delta_t = self.t-old_t
		norm = self.norm()
		lyap = log(norm) / delta_t
		super(jitcode_transversal_lyap, self).set_initial_value(self._y, self.t)
		
		return self._y[self.G.main_indices], lyap

class jitcode_restricted_lyap(jitcode_lyap):
	"""
	Calculates the largest Lyapunov exponent in orthogonal direction to a predefined plane, i.e. the projection of the tangent vector onto that plane vanishes. See `this test <https://github.com/neurophysik/jitcode/blob/master/tests/test_restricted_lyap.py>`_ for an example of usage. The handling is the same as that for `jitcode_lyap` except for:
	
	Parameters
	----------
	vectors : iterable of pairs of NumPy arrays
		A basis of the plane, whose projection shall be removed.
	"""
	
	def __init__(self, f_sym=(), vectors=[], **kwargs):
		kwargs["n_lyap"] = 1
		super(jitcode_restricted_lyap,self).__init__(f_sym,**kwargs)
		self.vectors = [ vector/np.linalg.norm(vector) for vector in vectors ]
	
	def norms(self):
		n = self.n_basic
		tangent_vector = self._y[n:2*n]
		for vector in self.vectors:
			tangent_vector -= np.dot(vector, tangent_vector)*vector
		norm = np.linalg.norm(tangent_vector)
		tangent_vector /= norm
		if not np.isfinite(norm):
			warn("Norm of perturbation vector for Lyapunov exponent out of numerical bounds. You probably waited too long before renormalising and should call integrate with smaller intervals between steps (as renormalisations happen once with every call of integrate).")
		return norm, tangent_vector


