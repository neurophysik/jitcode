from itertools import permutations

from symengine import Symbol

from jitcode import jitcode


p = Symbol("p")

f = [1/p]

for integrator in ["dopri5","RK45"]:
	def set_integrator(ODE):
		ODE.set_integrator(integrator)
	
	def set_parameters(ODE):
		ODE.set_parameters(1)
	
	def set_initial_value(ODE):
		ODE.set_initial_value([0], 0)
	
	at_east_one_working_order = False
	for functions in permutations([set_integrator,set_parameters,set_initial_value]):
		ODE = jitcode(f, control_pars=[p], verbose=False)
		try:
			for function in functions:
				function(ODE)
		except RuntimeError as exception:
			assert str(exception).startswith("Something needs parameters to be set")
		else:
			assert abs(ODE.integrate(1.)-1) < 1e-8
			at_east_one_working_order = True
		
		print( ".", end="", flush=True )
	assert at_east_one_working_order

f_2 = [1]

for integrator in ["dopri5","RK45"]:
	def set_integrator(ODE):
		ODE.set_integrator(integrator)
	
	def set_initial_value(ODE):
		ODE.set_initial_value([0], 0)
	
	at_east_one_working_order = False
	for functions in permutations([set_integrator,set_initial_value]):
		ODE = jitcode(f_2, verbose=False)
		try:
			for function in functions:
				function(ODE)
		except RuntimeError as exception:
			assert str(exception).startswith("Something needs parameters to be set")
		else:
			assert abs(ODE.integrate(1.)-1) < 1e-8
			at_east_one_working_order = True
		
		print( ".", end="", flush=True )
	assert at_east_one_working_order

print("")
