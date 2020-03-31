from inspect import signature

from scipy.integrate import ode
from scipy.integrate._ode import find_integrator
from scipy.integrate._ivp.ivp import METHODS as ivp_methods
from numpy import inf

class UnsuccessfulIntegration(Exception):
	"""
		This exception is raised when the integrator cannot meet the accuracy and step-size requirements. If you want to know the exact state of your system before the integration fails or similar, catch this exception.
	"""
	pass

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
	This is a wrapper around the integrators from scipy.integrate.solve_ivp making them work like scipy.integrate.ode (mostly).
	"""
	
	def __init__(self,name,f,jac=None,**kwargs):
		info = integrator_info(name)
		self.ivp_class = info["integrator"]
		self.f = f
		self.jac = jac
		self.wants_jac = info["wants_jac"]
		
		# Dictionary to be passed as arguments to the integrator and store stuff
		self.kwargs = {
				"t_bound": inf,
				"vectorized": False,
				"fun": self.f
			}
		if self.wants_jac:
			self.kwargs["jac"] = self.jac
		self.kwargs.update(kwargs)
	
	def set_integrator(self,*args,**kwargs):
		raise AssertionError("This method should not be called")
	
	@property
	def _y(self):
		return self.kwargs["y0"]
	
	@property
	def t(self):
		return self.kwargs["t0"]
	
	def try_to_initiate(self):
		"""
		initiate the integrator if all required arguments have been set
		"""
		if (
				"t0" in self.kwargs and
				"y0" in self.kwargs
			):
			self.backend = self.ivp_class(**self.kwargs)
	
	def set_initial_value(self, initial_value, time=0.0):
		self.kwargs["t0"] = time
		self.kwargs["y0"] = initial_value
		self.try_to_initiate()
	
	def set_params(self,*args):
		raise NotImplementedError("This method should not be called anymore")
	
	def integrate(self,t):
		while self.backend.t < t or self.backend.t_old is None:
			self.backend.step()
			if self.backend.status == "failed":
				raise UnsuccessfulIntegration
		self.kwargs["y0"] = self.backend.dense_output()(t)
		self.kwargs["t0"] = t
		return self.kwargs["y0"]
	
	def successful(self):
		return self.backend.status != "failed"

class IVP_wrapper_no_interpolation(IVP_wrapper):
	def integrate(self,t):
		if self.backend.t < t:
			# Manually un-finish the integrator:
			self.backend.t_bound = t
			self.backend.status = "running"
			
			while self.backend.status == "running":
				self.backend.step()
			
			self.kwargs["y0"] = self.backend.y
			self.kwargs["t0"] = t
			if self.backend.status == "failed":
				raise UnsuccessfulIntegration
		
		elif self.backend.t > t:
			raise ValueError("Target time smaller than current time. Cannot integrate backwards in time")
		
		return self.kwargs["y0"]

class ODE_wrapper(ode):
	"""
	This is a wrapper around Scipy’s ODE that mainly avoids confusing and pointless error messages and throws proper exceptions.
	"""
	def integrate(self,t,step=False,relax=False):
		if t>self.t or step or relax:
			result = super(ODE_wrapper,self).integrate(t,step,relax)
			if self.successful():
				return result
			else:
				raise UnsuccessfulIntegration
		elif t==self.t:
			return self._y
		else:
			raise ValueError("Target time smaller than current time. Cannot integrate backwards in time")
	
	@property
	def params(self):
		return self.f_params
	
	def set_params(self,*args):
		raise NotImplementedError("This method should not be called anymore")

class empty_integrator(object):
	"""
	This is a dummy class that mimicks some basic properties of scipy.integrate.ode or the above wrappers, respectively. It exists to store states and parameters and to raise exceptions in the same interface.
	"""

	def __init__(self):
		self.params = ()
		self._y = []
		self._t = None
	
	@property
	def t(self):
		if self._t is None:
			raise RuntimeError("You must call set_initial_value first.")
		else:
			return self._t
	
	def set_integrator(self,*args,**kwargs):
		raise RuntimeError("This function must not be called")
	
	def set_initial_value(self, initial_value, time=0.0):
		self._y = initial_value
		self._t = time
	
	def set_params(self,*args):
		raise NotImplementedError("This method should not be called anymore")
	
	def integrate(self,*args,**kwargs):
		raise RuntimeError("You must call set_integrator first.")
	
	def successful(self):
		raise RuntimeError("You must call set_integrator first.")
	
