#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
For instance, we can calculate and print the Lyapunov exponents for the system from `example` as follows (changes highlighted):
"""

from __future__ import print_function


if __name__ == "__main__":
	# example-start
	from jitcode import jitcode_lyap, y
	from scipy.stats import sem
	import numpy as np
	
	a  = -0.025794
	b1 =  0.0065
	b2 =  0.0135
	c  =  0.02
	k  =  0.128
	
	f = [
		y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
		b1*y(0) - c*y(1),
		y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
		b2*y(2) - c*y(3)
		]
	
	initial_state = np.array([1.,2.,3.,4.])
	
	n = len(f)
	ODE = jitcode_lyap(f, n_lyap=n)
	ODE.set_integrator("vode")
	ODE.set_initial_value(initial_state,0.0)
	
	times = range(10,100000,10)
	lyaps = []
	for time in times:
		lyaps.append(ODE.integrate(time)[1])
	
	# converting to Numpy array for easier handling
	lyaps = np.vstack(lyaps)
	
	for i in range(n):
		lyap = np.average(lyaps[1000:,i])
		stderr = sem(lyaps[1000:,i]) # Note that this only an estimate
		print("%i. Lyapunov exponent: % .4f ± %.4f" % (i+1,lyap,stderr))
