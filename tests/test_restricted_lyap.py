#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcode import jitcode_restricted_lyap, y
import numpy as np
from scipy.stats import sem

a  = -0.025794
b  =  0.01
c  =  0.02

scenarios = [
		{"k": 0.128, "sign":-1},
		{"k":-0.128, "sign": 1},
		{"k": 0    , "sign": 0},
	]

for scenario in scenarios:
	f = [
		y(0)*(a-y(0))*(y(0)-1.0) - y(1) + scenario["k"]*(y(2)-y(0)),
		b*y(0) - c*y(1),
		y(2)*(a-y(2))*(y(2)-1.0) - y(3) + scenario["k"]*(y(0)-y(2)),
		b*y(2) - c*y(3)
		]
	
	if scenario["sign"]<0:
		initial_state = np.random.random(4)
	else:
		single = np.random.random(2)
		initial_state = np.hstack([single,single])
	
	vectors = [
		np.array([1.,0.,1.,0.]),
		np.array([0.,1.,0.,1.])
		]
	
	ODE = jitcode_restricted_lyap( f, vectors=vectors, verbose=False )
	ODE.set_integrator("dopri5")
	ODE.set_initial_value(initial_state,0.0)
	
	data = np.hstack(ODE.integrate(T)[1] for T in range(10,100000,10))
	
	lyap = np.average(data[500:])
	margin = sem(data[500:])
	sign = 0 if abs(lyap)<margin else np.sign(lyap)
	assert sign==scenario["sign"], "Test failed in scenario %s"%scenario
	print(".",end="",flush=True)

print("")
