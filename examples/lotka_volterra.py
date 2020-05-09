#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose, we want to implement the Lotka–Volterra model, which is described by the following equations:

.. math::
	
	\\begin{alignat*}{3}
	\\dot{B} &=&    γ · B &- φ · R · B\\\\
	\\dot{R} &=&\\, -ω · R &+ ν · R · B
	\\end{alignat*}

with :math:`γ = 0.6`, :math:`φ = 1.0`, :math:`ω = 0.5`, and :math:`ν = 0.5`.

We start with a few imports:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 1-2
	:dedent: 1

… and defining the control parameters:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 4-7
	:dedent: 1

The `y` that we imported from `jitcode` has to be used for the dynamical variables. However, to make our code use the same notation as the above equation, we can rename them:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 9
	:dedent: 1

We here might as well have written `R,B = y(0),y(1)`, but the above is more handy for larger systems. Note that the following code is written such that the order of our dynamical variables does not matter.

We implement the differential equation as a dictionary, mapping each of the dynamical variables to its derivative:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 11-14
	:dedent: 1

Note that there are other ways to define the derivative, e.g., used in the previous and following example.

Now, we have everything, we need to instantiate `jitcode`, i.e., to initialise the integration object:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 16
	:dedent: 1

Before we start integrating, we have to choose an integrator and define initial conditions. We here choose the 5th-order Dormand–Prince method. Moreover the initial :math:`B` shall be 0.5, the initial :math:`R` shall be 0.2, and the starting time should be :math:`t=0`.

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 17-19
	:dedent: 1

We then define an array of time points where we want to observe the system, and a dictionary of empty lists that shall be filled with our results.

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 21-22
	:dedent: 1

Finally, we perform the actual integration by calling `ODE.integrate` for each of our `times`. After this, we use `ODE.y_dict` to access the current state as a dictionary and appending its values to the respective lists.

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 23-26
	:dedent: 1

We can now plot or analyse our data, but that’s beyond the scope of JiTCODE. So we just save it:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:lines: 28
	:dedent: 1

Taking everything together, our code is:

.. literalinclude:: ../examples/lotka_volterra.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:linenos:
"""

if __name__ == "__main__":
	# example-start
	from jitcode import y, jitcode
	import numpy as np
	
	γ = 0.6
	φ = 1.0
	ω = 0.5
	ν = 0.5
	
	R,B = dynvars = [ y(i) for i in range(2) ]
	
	lotka_volterra_diff = {
			B:  γ*B - φ*R*B,
			R: -ω*R + ν*R*B,
		}
	
	ODE = jitcode(lotka_volterra_diff)
	ODE.set_integrator("dopri5")
	initial_state = { R: 0.2, B: 0.5 }
	ODE.set_initial_value(initial_state,0.0)
	
	times = np.arange(0.0,100,0.1)
	values = { R: [], B: [] }
	for time in times:
		ODE.integrate(time)
		for dynvar in dynvars:
			values[dynvar].append(ODE.y_dict[dynvar])
	
	np.savetxt("timeseries.dat",np.vstack((values[R],values[B])).T)
