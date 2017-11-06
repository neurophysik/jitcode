#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
For instance, let’s interpret the system from `example` as two oscillators (which is what it is), one consisting of the first and second and one of the third and fourth component. Furthermore, let’s change the control parameters a bit to make the two oscillators identical. We can then calculate the transversal Lyapunov exponents to the synchronisation manifold as follows (important changes are highlighted):

.. literalinclude:: ../examples/double_fhn_transversal_lyap.py
	:emphasize-lines: 6, 12, 14, 17, 19-20
	:start-after: example-st\u0061rt
	:dedent: 1
	:linenos:

Note that the initial state (line 17) is reduced in dimensionality and there is only one component for each synchronisation group.
"""

if __name__ == "__main__":
	# example-start
	from jitcode import jitcode_transversal_lyap, y
	from scipy.stats import sem
	import numpy as np
	
	a = -0.025794
	b =  0.01
	c =  0.02
	k =  0.128
	
	f = [
		y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
		b*y(0) - c*y(1),
		y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
		b*y(2) - c*y(3)
		]
	
	initial_state = np.array([1.,2.])
	
	groups = [ [0,2], [1,3] ]
	ODE = jitcode_transversal_lyap(f, groups=groups)
	ODE.set_integrator("lsoda")
	ODE.set_initial_value(initial_state,0.0)
	
	times = range(10,100000,10)
	lyaps = []
	for time in times:
		lyaps.append(ODE.integrate(time)[1])
	
	lyap = np.average(lyaps[1000:])
	stderr = sem(lyaps[1000:]) # Note that this only an estimate
	print("transversal Lyapunov exponent: % .4f ± %.4f" % (lyap,stderr))

