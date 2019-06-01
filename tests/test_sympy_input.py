"""
Tests whether things works independent of where symbols are imported from.
"""

import jitcode
import jitcode.sympy_symbols
import sympy
import symengine

symengine_manually = [
		symengine.Symbol("t",real=True),
		symengine.Function("y",real=True),
		symengine.cos,
	]

sympy_manually = [
		sympy.Symbol("t",real=True),
		sympy.Function("y",real=True),
		sympy.cos,
	]

jitcode_provisions = [
		jitcode.t,
		jitcode.y,
		symengine.cos,
	]

jitcode_sympy_provisions = [
		jitcode.sympy_symbols.t,
		jitcode.sympy_symbols.y,
		symengine.cos,
	]

mixed = [
		jitcode.sympy_symbols.t,
		jitcode.y,
		sympy.cos,
	]

results = set()

for t,y,cos in [
			symengine_manually,
			sympy_manually,
			jitcode_provisions,
			jitcode_sympy_provisions,
			mixed
		]:
	ODE = jitcode.jitcode( [cos(t)*y(0)], verbose=False )
	ODE.set_integrator("dopri5")
	ODE.set_initial_value([1.0],0.0)
	
	result = ODE.integrate(10)[0]
	results.add(result)

assert len(results)==1

