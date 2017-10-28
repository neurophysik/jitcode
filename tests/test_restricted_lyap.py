#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcode import jitcode_restricted_lyap, y
import numpy as np
from scipy.stats import sem
from symengine import Symbol

a = -0.025794
b =  0.01
c =  0.02
k = Symbol("k")

f = [
		y(0)*(a-y(0))*(y(0)-1.0) - y(1) + k*(y(2)-y(0)),
		b*y(0) - c*y(1),
		y(2)*(a-y(2))*(y(2)-1.0) - y(3) + k*(y(0)-y(2)),
		b*y(2) - c*y(3)
	]

vectors = [
		np.array([1.,0.,1.,0.]),
		np.array([0.,1.,0.,1.])
	]

ODE = jitcode_restricted_lyap(
		f,
		vectors = vectors,
		verbose = False,
		control_pars = [k]
	)
# Simplification would lead to trajectories diverging from the synchronisation manifold due to numerical noise.
ODE.generate_f_C(simplify=False)
ODE.set_integrator("dopri5")

scenarios = [
		{"k":-0.1, "sign": 1},
		{"k":-0.2, "sign": 1},
		{"k": 0.1, "sign":-1},
		{"k": 0.2, "sign":-1},
		{"k": 0  , "sign": 0},
	]

for scenario in scenarios:
	ODE.set_f_params(scenario["k"])
	
	if scenario["sign"]<0:
		initial_state = np.random.random(4)
	else:
		single = np.random.random(2)
		initial_state = np.hstack([single,single])
	ODE.set_initial_value(initial_state,0.0)
	
	times = range(10,100000,10)
	lyaps = np.hstack(ODE.integrate(time)[1] for time in times)
	
	# Check that we are still on the synchronisation manifold:
	assert ODE.y[0]==ODE.y[2]
	assert ODE.y[1]==ODE.y[3]
	
	Lyap = np.average(lyaps[500:])
	margin = sem(lyaps[500:])
	sign = np.sign(Lyap) if abs(Lyap)>margin else 0
	assert sign==scenario["sign"], "Test failed in scenario %s. (Lyapunov exponent: %f±%f)" % (scenario,Lyap,margin)
	print(".",end="",flush=True)

print("")

