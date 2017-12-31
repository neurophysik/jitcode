import numpy as np
from numpy.random import choice, uniform
from time import process_time
from scipy.integrate import ode, solve_ivp, odeint
from scipy.integrate._ivp.ivp import METHODS
from jitcode import jitcode, y
from symengine import sin

solver_ode = "dopri5"
solver_ivp = "RK45"

# Context manager for timing
class timer(object):
	def __init__(self,name):
		self.name = name
	
	def __enter__(self):
		self.start = process_time()
	
	def __exit__(self,*args):
		end = process_time()
		duration = end-self.start
		print("%s took %.5f s" % (self.name,duration))

# The actual test
def test_scenario(name,fun,initial,times,rtol,atol):
	print(40*"-",name,40*"-",sep="\n")
	
	with timer("ode (%s)"%solver_ode):
		I = ode(fun)
		I.set_integrator(solver_ode,rtol=rtol,atol=atol,nsteps=10**8)
		I.set_initial_value(initial,0.0)
		result = np.vstack(I.integrate(time) for time in times)
	assert I.successful()
	
	inv_fun = lambda y,t: fun(t,y)
	with timer("odeint with suboptimal function (LSODA)"):
		result = odeint(
				func=inv_fun,
				y0=initial, t=[0.0]+list(times),
				rtol=rtol, atol=atol,
				mxstep=10**8
				)
	
	with timer("solve_ivp (%s) without result"%solver_ivp):
		I = solve_ivp(
				fun,
				t_span=(0,times[-1]),
				y0=initial,
				method=solver_ivp, rtol=rtol, atol=atol
			)
	assert I.status != -1
	
	with timer("solve_ivp (%s)"%solver_ivp):
		I = solve_ivp(
				fun,
				t_span=(0,times[-1]), t_eval=times,
				y0=initial,
				method=solver_ivp, rtol=rtol, atol=atol
			)
		result = I.y
	assert I.status != -1
	
	with timer("solve_ivp (%s) with dense_output"%solver_ivp):
		I = solve_ivp(
				fun,
				t_span=(0,times[-1]),
				y0=initial,
				method=solver_ivp, rtol=rtol, atol=atol,
				dense_output=True
			)
		result = np.vstack(I.sol(time) for time in times)
	assert I.status != -1
	
	with timer("%s with dense output"%solver_ivp):
		I = METHODS[solver_ivp](
				fun=fun,
				y0=initial, t0=0.0, t_bound=times[-1],
				rtol=rtol, atol=atol
			)
		def solutions():
			for time in times:
				while I.t < time:
					I.step()
				yield I.dense_output()(time)
		result = np.vstack(solutions())
	assert I.status != "failed"
	
	with timer("%s with manual resetting"%solver_ivp):
		I = METHODS[solver_ivp](
				fun=fun,
				y0=initial, t0=0.0, t_bound=times[-1],
				rtol=rtol, atol=atol
			)
		def solutions():
			for time in times:
				I.t_bound = time
				I.status = "running"
				while I.status == "running":
					I.step()
				yield I.y
		result = np.vstack(solutions())
	assert I.status != "failed"
	
	with timer("%s with reinitialising"%solver_ivp):
		def solutions():
			current_time = 0.0
			state = initial
			for time in times:
				I = METHODS[solver_ivp](
						fun=fun,
						y0=state, t0=current_time, t_bound=time,
						rtol=rtol, atol=atol
					)
				while I.status == "running":
					I.step()
				assert I.status != "failed"
				current_time = time
				state = I.y
				yield state
		result = np.vstack(solutions())

# Using compiled functions to make things faster
def get_compiled_function(f):
	dummy = jitcode(f,verbose=False)
	dummy.compile_C()
	return dummy.f

# The actual scenarios
test_scenario(
	name = "two coupled FitzHugh–Nagumo oscillators",
	fun = get_compiled_function([
			y(0)*(-0.025794-y(0))*(y(0)-1.0)-y(1)+0.128*(y(2)-y(0)),
			0.0065*y(0)-0.02*y(1),
			y(2)*(-0.025794-y(2))*(y(2)-1.0)-y(3)+0.128*(y(0)-y(2)),
			0.0135*y(2)-0.02*y(3)
		]),
	initial = np.array([1.,2.,3.,4.]),
	times = 2000+np.arange(0,100000,10),
	rtol = 1e-5,
	atol = 1e-8,
)


n, c, q = 100, 3.0, 0.2
A = choice( [1,0], size=(n,n), p=[q,1-q] )
omega = uniform(-0.5,0.5,n)
def kuramotos_f():
	for i in range(n):
		coupling_sum = sum(
				sin(y(j)-y(i))
				for j in range(n)
				if A[j,i]
			)
		yield omega[i] + c/(n-1)*coupling_sum

test_scenario(
	name = "random network of Kuramoto oscillators",
	fun = get_compiled_function(kuramotos_f),
	initial = uniform(0,2*np.pi,n),
	times = range(1,10001,10),
	rtol = 1e-13,
	atol = 1e-6,
)


