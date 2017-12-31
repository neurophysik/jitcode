from inspect import signature
from warnings import warn

from scipy.integrate._ode import find_integrator
from scipy.integrate._ivp.ivp import METHODS as ivp_methods
from numpy import inf

def integrator_info(name):
	"""
	Finds out the integrator from a given name, what backend it uses, and whether it can use a Jacobian.
	"""
	if name == 'zvode':
		raise NotImplementedError("JiTCODE does not natively support complex numbers yet.")
	
	if name in ivp_methods.keys():
		integrator = ivp_methods[name]
		return {
				"backend": "ivp",
				"wants_jac": "jac" in signature(integrator).parameters,
				"integrator": integrator
			}
	else:
		integrator = find_integrator(name)
		if integrator is None:
			raise RuntimeError("There is no integrator with that name; using fallback.")
		return {
				"backend": "ode",
				"wants_jac": "with_jacobian" in signature(integrator).parameters,
				"integrator": integrator
			}

class IVP_wrapper(object):
	"""
	This is a wrapper around the integrators from scipy.integrate.solve_ivp making them work like scipy.integrate.ode or raising errors when this is not possible.
	"""
	
	def __init__(self,name,f,jac=None,**kwargs):
		info = integrator_info(name)
		self.ivp_class = info["integrator"]
		
		self.kwargs = {
				"fun": f,
				"t_bound": inf,
				"vectorized": False,
			}
		self.kwargs.update(kwargs)
		
		if info["wants_jac"]:
			self.kwargs["jac"] = jac
		
		self.f_params = ()
		self.jac_params = ()
	
	def set_integrator(self,*args,**kwargs):
		raise AssertionError("This method should not be called")
	
	def set_initial_value(self, initial_value, time=0.0):
		self.t = time
		self.y = initial_value
		self.kwargs["t0"] = time
		self.kwargs["y0"] = initial_value
		self.backend = self.ivp_class(**self.kwargs)
	
	def set_f_params(self,*args):
		if args:
			raise NotImplementedError("The integrators from solve_ivp do not support setting control parameters at runtime yet.")
	
	def set_jac_params(self,*args):
		if args:
			raise NotImplementedError("The integrators from solve_ivp do not support setting control parameters at runtime yet.")
	
	def integrate(self,t,step=False,relax=False):
		while self.backend.t < t:
			self.backend.step()
		self._y = self.backend.dense_output()(t)
		self.t = t
		return self._y
	
	def successful(self):
		return self.backend.status != "failed"

class IVP_wrapper_no_interpolation(IVP_wrapper):
	def integrate(self,t,step=False,relax=False):
		self.backend.t_bound = t
		self.backend.status = "running"
		while self.backend.status == "running":
			self.backend.step()
		self._y = self.backend.y
		self.t = t
		return self._y

class empty_integrator(object):
	"""
	This is a dummy class that mimicks some basic properties of scipy.integrate.ode. It exists to store states and parameters and to raise exceptions in the same interface.
	"""

	def __init__(self):
		self.f_params = ()
		self.jac_params = ()
		self._y = []
		self._t = None
	
	@property
	def t(self):
		if self._t is None:
			raise RuntimeError("You must call set_integrator first.")
		else:
			return self._t
	
	def set_integrator(self,*args,**kwargs):
		raise AssertionError
	
	def set_initial_value(self, initial_value, time=0.0):
		self._y = initial_value
		self._t = time
	
	def set_f_params(self,*args):
		self.f_params = args
	
	def set_jac_params(self,*args):
		self.jac_params = args
	
	def integrate(self,t,step=False,relax=False):
		raise RuntimeError("You must call set_integrator first.")
	
	def successful(self):
		raise RuntimeError("You must call set_integrator first.")
	
