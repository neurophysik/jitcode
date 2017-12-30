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
	
