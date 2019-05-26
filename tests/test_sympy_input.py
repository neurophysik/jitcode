"""
Tests whether everything works with a pure SymPy input.
"""

from jitcode import jitcode
import sympy
import symengine

results = []

for backend in [sympy,symengine]:
	t = backend.Symbol("t", real=True)
	y = backend.Function("y")
	cos = backend.cos
	
	ODE = jitcode( [cos(t)*y(0)], verbose=False )
	ODE.set_integrator("dopri5")
	ODE.set_initial_value([1.0],0.0)
	
	result = ODE.integrate(10)[0]
	results.append(result)

assert results[0]==results[1]
