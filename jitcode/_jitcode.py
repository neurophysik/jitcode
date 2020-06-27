#!/usr/bin/python3
# -*- coding: utf-8 -*-

from warnings import warn
from types import FunctionType, BuiltinFunctionType
from inspect import signature
from itertools import count

from numpy import hstack, log
import numpy as np
import symengine

from jitcxde_common import jitcxde, checker
from jitcxde_common.helpers import sympify_helpers, sort_helpers, find_dependent_helpers
from jitcxde_common.numerical import random_direction, orthonormalise
from jitcxde_common.symbolic import collect_arguments, ordered_subs, replace_function
from jitcxde_common.transversal import GroupHandler

from jitcode.integrator_tools import empty_integrator, IVP_wrapper, IVP_wrapper_no_interpolation, ODE_wrapper, integrator_info, UnsuccessfulIntegration

#: the symbol for the state that must be used to define the differential equation. It is a function and the integer argument denotes the component. You may just as well define an analogous function directly with SymEngine or SymPy, but using this function is the best way to get the most of future versions of JiTCODE, in particular avoiding incompatibilities. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
y = symengine.Function("y")

#: the symbol for time for defining the differential equation. If your differential equation has no explicit time dependency (“autonomous system”), you do not need this. You may just as well define an analogous symbol directly with SymEngine or SymPy, but using this function is the best way to get the most of future versions of JiTCODE, in particular avoiding incompatibilities. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
t = symengine.Symbol("t", real=True)

def _is_C(function):
	return isinstance(function, BuiltinFunctionType)

def _is_lambda(function):
	return isinstance(function, FunctionType)

def _jac_from_f_with_helpers(f, helpers, simplify, n):
	dependent_helpers = [
			find_dependent_helpers(helpers,y(i))
			for i in range(n)
		]
	
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

class jitcode(jitcxde):
	"""
	Parameters
	----------
	f_sym : iterable of symbolic expressions or generator function yielding symbolic expressions or dictionary
		If an iterable or generator function, the `i`-th element is the `i`-th component of the value of the ODE’s derivative :math:`f(t,y)`. If a dictionary, it has to map the dynamical variables to its derivatives and the dynamical variables must be `y(0), y(1), …`.
	
	helpers : list of length-two iterables, each containing a symbol and a symbolic expression
		Each helper is a variable that will be calculated before evaluating the derivative and can be used in the latter’s computation. The first component of the tuple is the helper’s symbol as referenced in the derivative or other helpers, the second component describes how to compute it from `t`, `y` and other helpers. This is for example useful to realise a mean-field coupling, where the helper could look like `(mean, sum(y(i) for i in range(100))/100)`. (See `example_2` for an example.)
	
	wants_jacobian : boolean
		Tell JiTCODE to calculate and compile the Jacobian. For vanilla use, you do not need to bother about this as this is automatically set to `True` if the selected method of integration desires the Jacobian. However, it is sometimes useful if you want to manually apply some code-generation steps (e.g., to apply some tweaks).
		
	n : integer
		Length of `f_sym`. While JiTCODE can easily determine this itself (and will, if necessary), this may take some time if `f_sym` is a generator function and `n` is large. Take care that this value is correct – if it isn’t, you will not get a helpful error message.
	
	control_pars : iterable of symbols
		Each symbol corresponds to a control parameter that can be used when defining the equations and set after compilation using `set_parameters` (in the same order as given here). Using this makes sense if you need to do a parameter scan with short integrations for each parameter and you spend a considerable amount of time compiling.
	
	callback_functions : iterable
		Python functions that should be called at integration time (callback) when evaluating the derivative. Each element of the iterable represents one callback function as a tuple containing (in that order):
		
		*	A SymEngine function object used in `f_sym` to represent the function call. If you want to use any JiTCODE features that need the derivative, this must have a properly defined `f_diff` method with the derivative being another callback function (or constant).
		*	The Python function to be called. This function will receive the state array (`y`) as the first argument. All further arguments are whatever you use as arguments of the SymEngine function in `f_sym`. These can be any expression that you might use in the definition of the derivative and contain, e.g., dynamical variables, time, control parameters, and helpers. The only restriction is that the arguments are floats (and not vectors or similar). The return value must also be a float (or something castable to float). It is your responsibility to ensure that this function adheres to these criteria, is deterministic and sufficiently smooth with respect its arguments; expect nasty errors otherwise.
		*	The number of arguments, **excluding** the state array as mandatory first argument. This means if you have a variadic Python function, you cannot just call it with different numbers of arguments in `f_sym`, but you have to define separate callbacks for each of numer of arguments.
		
		See `this example <https://github.com/neurophysik/jitcdde/blob/master/examples/sunflower_callback.py>`_ (for JiTCDDE) for how to use this.
	
	verbose : boolean
		Whether JiTCODE shall give progress reports on the processing steps.
	
	module_location : string
		location of a module file from which functions are to be loaded (see `save_compiled`). If you use this, you need not give `f_sym` as an argument, but in this case you must give `n`. Depending on the arguments you provide, functionalities such as recompiling may not be available; but then the entire point of this option is to avoid these.
	"""
	
	dynvar = y
	
	def __init__(self,
				f_sym = (), *,
				helpers = None,
				wants_jacobian = False,
				n = None,
				control_pars = (),
				callback_functions = (),
				verbose = True,
				module_location = None,
			):
		jitcxde.__init__(self,n,verbose,module_location)
		
		self.f_sym = self._handle_input(f_sym)
		self._f_C_source = False
		self._jac_C_source = False
		self._helper_C_source = False
		self.helpers = sort_helpers(sympify_helpers(helpers or []))
		self.control_pars = control_pars
		self.control_par_values = ()
		self.callback_functions = callback_functions
		self._wants_jacobian = wants_jacobian
		
		self.integrator = empty_integrator()
		
		if self.jitced is None:
			self._initialise = None
			self.f = None
			self.jac = None
		else:
			# Load derivative and Jacobian if a compiled module has been provided
			self._initialise = self.jitced.initialise
			self.f = self.jitced.f
			self.jac = self.jitced.jac if hasattr(self.jitced,"jac") else None
		
		self._number_of_jac_helpers = None
		self._number_of_f_helpers = None
		
		self.general_subs = {
				control_par: symengine.Symbol("parameter_"+control_par.name)
				for control_par in self.control_pars
			}
	
	@checker
	def _check_non_empty(self):
		self._check_assert( self.f_sym(), "f_sym is empty." )
	
	@checker
	def _check_valid_arguments(self):
		for i,entry in enumerate(self.f_sym()):
			for argument in collect_arguments(entry,y):
				self._check_assert(
						argument[0] >= 0,
						"y is called with a negative argument (%i) in equation %i." % (argument[0],i),
					)
				self._check_assert(
						argument[0] < self.n,
						"y is called with an argument (%i) higher than the system’s dimension (%i) in equation %i." % (argument[0],self.n,i)
					)
	
	@checker
	def _check_valid_symbols(self):
		valid_symbols = [t] + [helper[0] for helper in self.helpers] + list(self.control_pars)
		
		for i,entry in enumerate(self.f_sym()):
			for symbol in entry.atoms(symengine.Symbol):
				self._check_assert(
						symbol in valid_symbols,
						"Invalid symbol (%s) in equation %i."  % (symbol.name,i),
					)
	
	@property
	def jac_sym(self):
		if not hasattr(self,"_jac_sym"):
			self.generate_jac_sym()
			self.report("generated symbolic Jacobian")
		return self._jac_sym
	
	def generate_jac_sym(self, simplify=True):
		"""
		generates the Jacobian using SymEngine’s differentiation.
		
		Parameters
		----------
		simplify : boolean
			Whether the resulting Jacobian should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`). This is almost always a good thing.
		"""
		self._jac_sym = _jac_from_f_with_helpers(self.f_sym, self.helpers, simplify, self.n)
	
	def _default_arguments(self):
		basics = [
				("t", "double const"),
				("Y", "PyArrayObject *__restrict const")
			]
		return basics
	
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
		
		# working copy
		f_sym_wc = (entry.subs(self.general_subs) for entry in self.f_sym())
		
		if simplify is None:
			simplify = self.n<=10
		if simplify:
			f_sym_wc = (entry.simplify(ratio=1) for entry in f_sym_wc)
		
		arguments = self._default_arguments()
		
		if self.helpers:
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
					arguments = arguments,
					omp = False,
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
			Whether a sparse Jacobian should be assumed for optimisation. Note that this does not mean that the Jacobian is stored, parsed or handled as a sparse matrix. This kind of optimisation would require `ode` or `solve_ivp` to be able to handle sparse matrices without structure in the sparseness.
		"""
		
		self._generate_helpers_C()
		
		# working copy
		jac_sym_wc = ( (entry.subs(self.general_subs) for entry in line) for line in self.jac_sym )
		self.sparse_jac = sparse
		
		arguments = self._default_arguments()
		if self.helpers:
			arguments.append(("general_helper","double const *__restrict const"))
		
		if do_cse:
			import sympy
			get_helper = sympy.Function("get_jac_helper")
			set_helper = symengine.Function("set_jac_helper")
			jac_sym_wc = sympy.Matrix([
					[ sympy.sympify(entry) for entry in line ]
					for line in jac_sym_wc
				])
			
			_cse = sympy.cse(
					sympy.sympify(jac_sym_wc),
					symbols = (get_helper(i) for i in count())
				)
			more_helpers = symengine.sympify(_cse[0])
			jac_sym_wc = symengine.sympify(_cse[1][0].tolist())
			
			if more_helpers:
				arguments.append(("jac_helper","double *__restrict const"))
				self.render_and_write_code(
						(set_helper(i, helper[1]) for i,helper in enumerate(more_helpers)),
						name = "jac_helpers",
						chunk_size = chunk_size,
						arguments = arguments,
						omp = False
					)
				self._number_of_jac_helpers = len(more_helpers)
		
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
					arguments = self._default_arguments() + [("general_helper","double *__restrict const")],
					omp = False,
				)
		
		self._helper_C_source = True
	
	def _compile_C(self):
		if (not _is_C(self.f)) or self._lacks_jacobian:
			self.compile_C()
			self.report("compiled C code")
	
	def compile_C(
			self,
			extra_compile_args=None, extra_link_args=None,
			verbose = False,
			modulename = None,
			omp = False,
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
		omp : pair of iterables of strings or boolean
			What compiler arguments shall be used for multiprocessing (using OpenMP). If `True`, they will be selected automatically. If empty or `False`, no compilation for multiprocessing will happen (unless you supply the relevant compiler arguments otherwise).
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
				control_pars = [par.name for par in self.control_pars],
				callbacks = [(fun.name,n_args) for fun,_,n_args in self.callback_functions],
			)
		
		self._compile_and_load(
				verbose,
				extra_compile_args, extra_link_args,
				omp
			)
		
		self.f = self.jitced.f
		if hasattr(self.jitced,"jac"):
			self.jac = self.jitced.jac
		self._initialise = self.jitced.initialise
	
	def _prepare_lambdas(self):
		if self.callback_functions:
			raise NotImplementedError("Callbacks do not work with lambdification. You must use the C backend.")
		
		if not hasattr(self,"_lambda_subs") or not hasattr(self,"_lambda_args"):
			if self.helpers:
				warn("Lambdification handles helpers by plugging them in. This may be very inefficient")
			
			self._lambda_subs = list(reversed(self.helpers))
			self._lambda_args = [t]
			for i in range(self.n):
				symbol = symengine.Symbol("dummy_argument_%i"%i)
				self._lambda_args.append(symbol)
				self._lambda_subs.append((y(i),symbol))
			self._lambda_args.extend(self.control_pars)
	
	def _generate_f_lambda(self):
		if not _is_lambda(self.f):
			self.generate_f_lambda()
			self.report("generated lambdified f")
	
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
		
		# working copy
		f_sym_wc = (ordered_subs(entry,self._lambda_subs) for entry in self.f_sym())
		
		if simplify is None:
			simplify = self.n<=10
		if simplify:
			f_sym_wc = (entry.simplify(ratio=1.0) for entry in f_sym_wc)
		
		lambdify = symengine.LambdifyCSE if do_cse else symengine.Lambdify
		core_f = lambdify(self._lambda_args,list(f_sym_wc))
		self.f = lambda t,Y: core_f(np.hstack([t,Y,self.control_par_values]))
		
		self.compile_attempt = False
	
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
		
		jac_matrix = symengine.Matrix([
				[ordered_subs(entry,self._lambda_subs) for entry in line]
				for line in self.jac_sym
			])
		
		lambdify = symengine.LambdifyCSE if do_cse else symengine.Lambdify
		core_jac = lambdify(self._lambda_args,jac_matrix)
		self.jac = lambda t,Y: core_jac(np.hstack([t,Y,self.control_par_values]))
		
		self.compile_attempt = False
	
	def generate_lambdas(self):
		"""
		If they do not already exists, this generates lambdified functions by calling `self.generate_f_lambda()` and, if wanted, `generate_jac_lambda()`.
		"""
		
		self._generate_f_lambda()
		if self._wants_jacobian:
			self._generate_jac_lambda()
		self.compile_attempt = False
	
	@property
	def _lacks_jacobian(self):
		return self._wants_jacobian and self.jac is None
	
	def generate_functions(self):
		"""
		The central function-generating function. Tries to compile the derivative and, if wanted, the Jacobian. If this fails, it generates lambdified functions as a fallback.
		"""
		
		if self.compile_attempt is None or self._lacks_jacobian:
			self._attempt_compilation(reset=False)
		
		if not self.compile_attempt:
			self.generate_lambdas()
	
	@property
	def t(self):
		return self.integrator.t
	
	@property
	def _y(self):
		return self.integrator._y
	
	@property
	def y(self):
		return self.integrator._y
	
	@property
	def y_dict(self):
		"""
		The current state of the system as a dictionary mapping dynamical variables to their current value.
		Note that if you use this often, you may want to use `self.y` instead for efficiency.
		"""
		
		return { self.dynvar(i):self.y[i] for i in range(self.n) }
	
	def set_initial_value(self, initial_value, time=0.0):
		"""
		Same as the analogous function in SciPy’s ODE, except that it also accepts the initial_value in form of a dictionary that maps dynamical variables to their initial value.
		"""
		
		if isinstance(initial_value,dict):
			initial_value = self._list_from_dynvar_dict(
					initial_value,
					"initial value",
					self.n,
				)
		
		if self.n != len(initial_value):
			raise ValueError("The dimension of the initial value does not match the dimension of your differential equations.")
		
		self.integrator.set_initial_value(initial_value, time)
	
	def set_integrator(self,name,nsteps=10**6,interpolate=True,**integrator_params):
		"""
		Analogous to the function in SciPy’s ODE with the same name. This automatically generates the derivative and Jacobian if they do not exist yet and are needed. You can also choose integrators from `scipy.integrate.solve_ivp`.
		
		Parameters
		----------
		name: name of the integrator
			One of the following (or a new method supported by either backend):
			
			* `"dopri5"` – Dormand’s and Prince’s explicit fifth-order method via `ode`
			* `"RK45"` – Dormand’s and Prince’s explicit fifth-order method via `solve_ivp`
			* `"dop853"` – DoP853 (explicit) via `ode`
			* `"RK23"` – Bogacki’s and Shampine’s explicit third-order method via `solve_ivp`
			* `"BDF"` – Implicit backward-differentiation formula via `solve_ivp`
			* `"lsoda"` – LSODA (implicit) via `ode`
			* `"LSODA"` – LSODA (implicit) via `solve_ivp`
			* `"Radau"` – The implicit Radau method via `solve_ivp`
			* `"vode"` – VODE (implicit) via `ode`
			
			The `solve_ivp` methods are usually slightly faster for large differential equations, but they come with a massive overhead that makes them considerably slower for small differential equations. Implicit solvers are slower than explicit ones, except for stiff problems. If you don’t know what to choose, start with `"dopri5"`.
		
		nsteps: integer
 			Same as the respective parameter of the `ode` solvers, but with a higher default value to avoid annoying errors when getting rid of transients.
		
		interpolate: boolean
			Whether the sampled solutions for `solve_ivp` solvers shall be obtained using interpolation. If your sampling step is small, this may make things faster; otherwise it depends. This may also make the results slightly less accurate.
		
		integrator_params
			Parameters passed to the respective integrator. See its documentation for more.
		"""
		
		info = integrator_info(name)
		self._wants_jacobian |= info["wants_jac"]
		
		old_integrator = self.integrator
		
		self.generate_functions()
		
		if info["backend"] == "ode":
			self.integrator = ODE_wrapper(self.f,self.jac)
			self.integrator.set_integrator(
					name,
					nsteps = nsteps,
					**integrator_params
				)
		elif info["backend"] == "ivp":
			if not interpolate and name=="LSODA":
				raise NotImplementedError("LSODA doesn’t work without interpolation.")
			IVP = IVP_wrapper if interpolate else IVP_wrapper_no_interpolation
			self.integrator = IVP(
					name,
					self.f,
					self.jac,
					**integrator_params
				)
		
		# Restore state and params, if applicable:
		try:
			self.set_initial_value(old_integrator._y,old_integrator.t)
		except (AttributeError,RuntimeError):
			pass
	
	def initialise(self):
		if self._initialise is not None:
			self._initialise(
					*self.control_par_values,
					*[callback for _,callback,_ in self.callback_functions]
				)
	
	def set_parameters(self,*args):
		"""
		Same as `set_f_params` and `set_jac_params` for SciPy’s ODE (both  sets of parameters are set simultaneuosly, because they should be the same anyway).
		
		The parameters can be passed as different arguments or as a list or other sequence.
		"""
		try:
			self.control_par_values = tuple(args[0])
		except (TypeError,IndexError):
			self.control_par_values = args
		else:
			if len(args)>1:
				raise TypeError("Argument must either be a single sequence or multiple numbers.")
		
		self.initialise()
	
	def set_f_params(self, *args):
		warn("This function has been replaced by `set_parameters`")
		self.set_parameters(*args)
	
	def set_jac_params(self, *args):
		warn("This function has been replaced by `set_parameters`")
		self.set_parameters(*args)
	
	def integrate(self,*args,**kwargs):
		return self.integrator.integrate(*args,**kwargs)
	
	def successful(self):
		raise NotImplementedError("JiTCODE doesn’t support ODE’s `successful` because it’s a very archaic and tedious way of handling things. Instead the exception `UnsuccessfulIntegration` is raised in case of an unsuccessful integration. If you want your program to do anything other than aborting in case of an unsuccessful integration, catch `UnsuccessfulIntegration` and act upon it as you deem fit.")

class jitcode_lyap(jitcode):
	"""
	the handling is the same as that for `jitcode` except for:
	
	Parameters
	----------
	n_lyap : integer
		Number of Lyapunov exponents to calculate. If negative or larger than the dimension of the system, all Lyapunov exponents are calculated.
	
	simplify : boolean
		Whether the differential equations for the tangent vector shall be subjected to SymEngine’s `simplify`. Doing so may speed up the time evolution but may slow down the generation of the code (considerably for large differential equations). If `None`, this will be automatically disabled for `n>10`.
	"""
	
	def __init__( self, f_sym=(), n_lyap=-1, simplify=None, **kwargs ):
		self.n_basic = kwargs.pop("n",None)
		
		f_basic = self._handle_input(f_sym,n_basic=True)
		self._n_lyap = n_lyap if (0<=n_lyap<=self.n_basic) else self.n_basic
		if simplify is None:
			simplify = self.n_basic<=10
		
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
	
	def set_initial_value(self, y, time=0.0):
		if isinstance(y,dict):
			y = self._list_from_dynvar_dict(y,"initial value",self.n_basic)
		
		new_y = [y]
		for _ in range(self._n_lyap):
			new_y.append(random_direction(self.n_basic))
		
		super(jitcode_lyap, self).set_initial_value(hstack(new_y), time)
	
	def set_integrator(self,name,interpolate=None,**kwargs):
		"""
		Same as for `jitcode` except that `interpolate` defaults to `False` for RK45 and Radau to avoid an accumulation error through the nature of the interpolation algorithm. Using `LSODA` as an integrator is discouraged.
		"""
		if interpolate is None:
			interpolate = name in ["RK45","Radau"]
		if name == "LSODA":
			warn("Using LSODA for Lyapunov exponents is discouraged since interpolation errors may accumulate.")
		super(jitcode_lyap,self).set_integrator(name,interpolate,**kwargs)
	
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
	
	@property
	def y_dict(self):
		return { self.dynvar(i):self.y[i] for i in range(self.n_basic) }


class jitcode_transversal_lyap(jitcode,GroupHandler):
	"""
	Calculates the largest Lyapunov exponent in orthogonal direction to a predefined synchronisation manifold, i.e. the projection of the tangent vector onto that manifold vanishes. In contrast to `jitcode_restricted_lyap`, this performs some transformations tailored to this specific application that may strongly reduce the number of differential equations and ensure a dynamics on the synchronisation manifold.
	
	The handling is the same as that for `jitcode` except for:
	
	Parameters
	----------
	groups : iterable of iterables of integers
		each group is an iterable of indices that identify dynamical variables that are synchronised on the synchronisation manifold.
	
	simplify : boolean
		Whether the transformed differential equations shall be subjected to SymEngine’s `simplify`. Doing so may speed up the time evolution but may slow down the generation of the code (considerably for large differential equations). If `None`, this will be automatically disabled for `n>10`.
	"""
	
	# Read the accompanying paper to understand the internals of this.
	
	def __init__( self, f_sym=(), groups=(), simplify=None, **kwargs ):
		GroupHandler.__init__(self,groups)
		self.n = kwargs.pop("n",None)
		
		f_basic,extracted = self.extract_main(self._handle_input(f_sym))
		helpers = sort_helpers(sympify_helpers( kwargs.pop("helpers",[]) ))
		
		if simplify is None:
			simplify = self.n<=10
		
		z = symengine.Function("z")
		z_vector = [z(i) for i in range(self.n)]
		tangent_vector = self.back_transform(z_vector)
		
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
				y(i): z(self.map_to_main(i))
				for i in range(self.n)
			}
		
		def finalise(entry):
			entry = entry.subs(substitutions)
			if simplify:
				entry = entry.simplify(ratio=1)
			return replace_function(entry,z,y)
		
		def f_lyap():
			for entry in self.iterate(tangent_vector_f()):
				if type(entry)==int:
					yield finalise(extracted[self.main_indices[entry]])
				else:
					yield finalise(entry[0]-entry[1])
		
		helpers = ((helper[0],finalise(helper[1])) for helper in helpers)
		
		super(jitcode_transversal_lyap, self).__init__(
				f_lyap,
				helpers = helpers,
				n = self.n,
				**kwargs
			)
	
	def set_initial_value(self, y, time=0.0):
		"""
		Like the analogous function of `jitcode`/`scipy.integrate.ode`, except that only one initial value per group of synchronised components has to be provided (in the same order as the `groups` argument of the constructor).
		"""
		if isinstance(y,dict):
			raise NotImplementedError("Dictionary input is not supported for transversal Lyapunov exponents")
		assert len(y)==len(self.groups), "Initial state too long. Provide only one value per synchronisation group"
		
		new_y = np.empty(self.n)
		new_y[self.main_indices] = y
		new_y[self.tangent_indices] = random_direction(len(self.tangent_indices))
		super(jitcode_transversal_lyap, self).set_initial_value(new_y, time)
	
	def set_integrator(self,name,interpolate=False,**kwargs):
		"""
		Same as for `jitcode` except that `interpolate` defaults to `False` for RK45 and Radau to avoid an accumulation error through the nature of the interpolation algorithm. Using `LSODA` as an integrator is discouraged.
		"""
		if interpolate is None:
			interpolate = name in ["RK45","Radau"]
		if name == "LSODA":
			warn("Using LSODA for Lyapunov exponents is discouraged since interpolation errors may accumulate.")
		super(jitcode_transversal_lyap,self).set_integrator(name,interpolate,**kwargs)
	
	def norm(self):
		tangent_vector = self._y[self.tangent_indices]
		norm = np.linalg.norm(tangent_vector)
		tangent_vector /= norm
		if not np.isfinite(norm):
			warn("Norm of perturbation vector for Lyapunov exponent out of numerical bounds. You probably waited too long before renormalising and should call integrate with smaller intervals between steps (as renormalisations happen once with every call of integrate).")
		self._y[self.tangent_indices] = tangent_vector
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
		
		return self._y[self.main_indices], lyap
	
	@property
	def y_dict(self):
		raise NotImplementedError("Dictionary output is not supported for transversal Lyapunov exponents")


class jitcode_restricted_lyap(jitcode_lyap):
	"""
	Calculates the largest Lyapunov exponent in orthogonal direction to a predefined plane, i.e. the projection of the tangent vector onto that plane vanishes. See `this test <https://github.com/neurophysik/jitcode/blob/master/tests/test_restricted_lyap.py>`_ for an example of usage. The handling is the same as that for `jitcode_lyap` except for:
	
	Parameters
	----------
	vectors : iterable of pairs of NumPy arrays
		A basis of the plane, whose projection shall be removed.
	"""
	
	def __init__(self, f_sym=(), vectors=(), **kwargs):
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

def test(omp=True,sympy=True):
	"""
		Runs a quick simulation to test whether:
		
		* a compiler is available and can be interfaced by Setuptools,
		* OMP libraries are available and can be assessed,
		* SymPy is available.
		
		The latter two tests can be deactivated with the respective argument. This is not a full software test but rather a quick sanity check of your installation. If successful, this function just finishes without any message.
	"""
	if sympy:
		import sympy
	ODE = jitcode( [y(1),-y(0)], verbose=False )
	ODE.generate_f_C(chunk_size=1)
	ODE.compile_C(omp=omp)
	ODE.set_integrator("dopri5")
	ODE.set_initial_value([1,2])
	ODE.integrate(0.1)
	
