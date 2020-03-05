#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Integration test of jitcode_restricted_lyap and jitcode_transversal_lyap by comparing them to each other.
"""

from itertools import combinations
from jitcode import jitcode_restricted_lyap, y, jitcode_transversal_lyap
import numpy as np
from scipy.stats import sem
from symengine import Symbol

a = -0.025794
b =  0.01
c =  0.02
k = Symbol("k")

scenarios = [
	{
	"f":
		[
			y(0)*(a-y(0))*(y(0)-1.0) - y(3) + k*(y(1)-y(0)),
			y(1)*(a-y(1))*(y(1)-1.0) - y(4) + k*(y(2)-y(1)),
			y(2)*(a-y(2))*(y(2)-1.0) - y(5) + k*(y(0)-y(2)),
			b*y(0) - c*y(3),
			b*y(1) - c*y(4),
			b*y(2) - c*y(5),
		],
	"vectors":
		[
			[1.,1.,1.,0.,0.,0.],
			[0.,0.,0.,1.,1.,1.]
		],
	"groups":
		( [0,1,2], [3,4,5] )
	},
	{
	"f":
		[
			y(0)*(a-y(0))*(y(0)-1.0) - y(1) + k*(y(2)-y(0)),
			b*y(0) - c*y(1),
			y(2)*(a-y(2))*(y(2)-1.0) - y(3) + k*(y(4)-y(2)),
			b*y(2) - c*y(3),
			y(4)*(a-y(4))*(y(4)-1.0) - y(5) + k*(y(0)-y(4)),
			b*y(4) - c*y(5),
		],
	"vectors":
		[
			[1.,0.,1.,0.,1.,0.],
			[0.,1.,0.,1.,0.,1.]
		],
	"groups":
		( [0,2,4], [1,3,5] )
	},
]

couplings = [
		{"k":-0.1, "sign": 1},
		{"k":-0.2, "sign": 1},
		{"k": 0.1, "sign":-1},
		{"k": 0.2, "sign":-1},
		{"k": 0  , "sign": 0},
	]

for scenario in scenarios:
	n = len(scenario["f"])
	
	ODE1 = jitcode_restricted_lyap(
			scenario["f"],
			vectors = scenario["vectors"],
			verbose = False,
			control_pars = [k]
		)
	# Simplification would lead to trajectories diverging from the synchronisation manifold due to numerical noise.
	ODE1.generate_f_C(simplify=False)
	ODE1.set_integrator("dopri5")

	ODE2 = jitcode_transversal_lyap(
			scenario["f"],
			groups = scenario["groups"],
			verbose = False,
			control_pars = [k]
		)
	ODE2.set_integrator("dopri5")
	
	for coupling in couplings:
		ODE1.set_parameters(coupling["k"])
		ODE2.set_parameters(coupling["k"])
		
		if coupling["sign"]<0:
			initial_state = np.random.random(n)
		else:
			single = np.random.random(2)
			initial_state = np.empty(n)
			for j,group in enumerate(scenario["groups"]):
				for i in group:
					initial_state[i] = single[j]
		ODE1.set_initial_value(initial_state,0.0)
		
		ODE2.set_initial_value(np.random.random(2),0.0)
		
		times = range(100,100000,100)
		lyaps1 = np.hstack([ODE1.integrate(time)[1] for time in times])
		lyaps2 = np.hstack([ODE2.integrate(time)[1] for time in times])
		
		# Check that we are still on the synchronisation manifold:
		for group in scenario["groups"]:
			for i,j in combinations(group,2):
				assert ODE1.y[i]==ODE1.y[j], "If this fails, the test is broken, not JiTCODE itself."
		
		Lyap1 = np.average(lyaps1[500:])
		Lyap2 = np.average(lyaps2[500:])
		margin1 = sem(lyaps1[500:])
		margin2 = sem(lyaps2[500:])
		sign1 = np.sign(Lyap1) if abs(Lyap1)>margin1 else 0
		sign2 = np.sign(Lyap2) if abs(Lyap2)>margin2 else 0
		assert sign1==coupling["sign"]
		assert sign2==coupling["sign"]
		assert abs(Lyap1-Lyap2)<max(margin1,margin2), "%f±%f \t %f±%f"%(Lyap1,margin1,Lyap2,margin2)
		print( ".", end="", flush=True )

print("")
