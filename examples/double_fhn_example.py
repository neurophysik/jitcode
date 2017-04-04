#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose our differential equation is :math:`\dot{y} = f(y)` with :math:`y∈ℝ^4`,

.. math::

	f(y) = \\left(
	\\begin{matrix}
	y_0 ( a-y_0 ) ( y_0-1) - y_1 + k (y_2 - y_0) \\\\
	b_1 y_0 - c y_1 \\\\
	y_2 ( a-y_2 ) ( y_2-1 ) - y_3 + k (y_0 - y_2)\\\\
	b_2 y_2 - c y_3
	\\end{matrix} \\right),

and :math:`a = -0.025794`, :math:`b_1 = 0.0065`, :math:`b_2 = 0.0135`, :math:`c = 0.02`, and :math:`k = 0.128`.
Then the following code integrates the above for 100000 time units, with :math:`y(t=0) = (1,2,3,4)`, and writes the results to :code:`timeseries.dat`:
"""

if __name__ == "__main__":
	# example-start
	from jitcode import jitcode, y
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
	
	ODE = jitcode(f)
	ODE.set_integrator("dopri5")
	ODE.set_initial_value(initial_state,0.0)
	
	times = range(10,100000,10)
	data = []
	for time in times:
		data.append(ODE.integrate(time))
	
	np.savetxt("timeseries.dat", data)
