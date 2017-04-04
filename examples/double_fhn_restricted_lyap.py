#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcode import jitcode_restricted_lyap, y
import numpy as np
from scipy.stats import sem

a  = -0.025794
b1 =  0.01
b2 =  0.01
c  =  0.02
k  =  0.128

f = [
	y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
	b1*y(0) - c*y(1),
	y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
	b2*y(2) - c*y(3)
	]

initial_state = np.random.random(4)

vectors = [
	np.array([1.,0.,1.,0.]),
	np.array([0.,1.,0.,1.])
	]

ODE = jitcode_restricted_lyap(f, vectors=vectors)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state,0.0)

data = np.hstack(ODE.integrate(T)[1] for T in range(10,100000,10))

print(np.average(data[500:]), sem(data[500:]))
