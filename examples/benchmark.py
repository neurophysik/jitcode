from time import process_time

import numpy as np
from scipy.integrate import ode, odeint, solve_ivp
from scipy.integrate._ivp.ivp import METHODS
from symengine import sin

from jitcode import jitcode, y


solver_ode = "dopri5"
solver_ivp = "RK45"

# Context manager for timing
class timer:
	def __init__(self,name):
		self.name = name
	
	def __enter__(self):
		self.start = process_time()
	
	def __exit__(self,*args):
		end = process_time()
		duration = end-self.start
		print(f"{self.name} took {duration:.5f}s")

# The actual test
def test_scenario(name,fun,initial,times,rtol,atol):
	print(40*"-",name,40*"-",sep="\n")
	
	with timer(f"ode ({solver_ode})"):
		solver = ode(fun)
		solver.set_integrator(solver_ode,rtol=rtol,atol=atol,nsteps=10**8)
		solver.set_initial_value(initial,0.0)
		_result = np.vstack([solver.integrate(time) for time in times])
	assert solver.successful()
	
	inv_fun = lambda y,t: fun(t,y)
	with timer("odeint with suboptimal function (LSODA)"):
		_result = odeint(
				func=inv_fun,
				y0=initial, t=[0.0, *times],
				rtol=rtol, atol=atol,
				mxstep=10**8
				)
	
	with timer(f"solve_ivp ({solver_ivp}) without result"):
		solver = solve_ivp(
				fun,
				t_span=(0,times[-1]),
				y0=initial,
				method=solver_ivp, rtol=rtol, atol=atol
			)
	assert solver.status != -1
	
	with timer(f"solve_ivp ({solver_ivp})"):
		solver = solve_ivp(
				fun,
				t_span=(0,times[-1]), t_eval=times,
				y0=initial,
				method=solver_ivp, rtol=rtol, atol=atol
			)
		_result = solver.y
	assert solver.status != -1
	
	with timer(f"solve_ivp ({solver_ivp}) with dense_output"):
		solver = solve_ivp(
				fun,
				t_span=(0,times[-1]),
				y0=initial,
				method=solver_ivp, rtol=rtol, atol=atol,
				dense_output=True
			)
		_result = np.vstack([solver.sol(time) for time in times])
	assert solver.status != -1
	
	with timer(f"{solver_ivp} with dense output"):
		solver = METHODS[solver_ivp](
				fun=fun,
				y0=initial, t0=0.0, t_bound=times[-1],
				rtol=rtol, atol=atol
			)
		def solutions():
			for time in times:
				while solver.t < time:
					solver.step()
				yield solver.dense_output()(time)
		_result = np.vstack(list(solutions()))
	assert solver.status != "failed"
	
	with timer(f"{solver_ivp} with manual resetting"):
		solver = METHODS[solver_ivp](
				fun=fun,
				y0=initial, t0=0.0, t_bound=times[-1],
				rtol=rtol, atol=atol
			)
		def solutions():
			for time in times:
				solver.t_bound = time
				solver.status = "running"
				while solver.status == "running":
					solver.step()
				yield solver.y
		_result = np.vstack(list(solutions()))
	assert solver.status != "failed"
	
	with timer(f"{solver_ivp} with reinitialising"):
		def solutions():
			current_time = 0.0
			state = initial
			for time in times:
				solver = METHODS[solver_ivp](
						fun=fun,
						y0=state, t0=current_time, t_bound=time,
						rtol=rtol, atol=atol
					)
				while solver.status == "running":
					solver.step()
				assert solver.status != "failed"
				current_time = time
				state = solver.y
				yield state
		_result = np.vstack(list(solutions()))

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


rng = np.random.default_rng(seed=42)

n, c, q = 100, 3.0, 0.2
A = rng.choice( [1,0], size=(n,n), p=[q,1-q] )
omega = rng.uniform(-0.5,0.5,n)

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
	initial = rng.uniform(0,2*np.pi,n),
	times = range(1,10001,10),
	rtol = 1e-13,
	atol = 1e-6,
)
