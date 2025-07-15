#!/usr/bin/python3

"""
This example showcases several advanced features of JiTCODE that are relevant for an efficient integration of more complex systems as well as how to deal with some special situations. Therefore it is pretty bizarre from a dynamical perspective.

Suppose we want to integrate a system of :math:`N=500` Rössler oscillators, with the :math:`i`-th oscillator being described by the following differential equations (note that we used :math:`v` instead of :math:`y` for the second dynamical variable to avoid overloading symbols):

.. math::
	\\begin{alignedat}{1}
	\\dot{x}_i &= -ω_i v_i - z_i + k \\sin(t) \\sum_{j=0}^N A_{ij} (x_j-x_i) \\\\
	\\dot{v}_i &= ω_i x_i + a v_i \\\\
	\\dot{z}_i &= b + z_i (x_i -c) + k \\sum_{j=0}^N (z_j-z_i)
	\\end{alignedat}

The control parameters shall be :math:`a = 0.165`, :math:`b = 0.2`, :math:`c = 10.0`, and :math:`k = 0.01`. The (frequency) parameter :math:`ω_i` shall be picked randomly from the uniform distribution on :math:`[0.8,1.0]` for each :math:`i`. :math:`A∈ℝ^{N×N}` shall be the adjacency matrix of a one-dimensional small-world network (which shall be provided by a function `small_world_network` in the following example code). So, the :math:`x` components are coupled diffusively with a small-world coupling topology, while the :math:`z` components are coupled diffusively to their mean field, with the coupling term being modulated with :math:`\\sin(t)`.

Without further ado, here is the example code (`complete running example <https://raw.githubusercontent.com/neurophysik/jitcode/master/examples/SW_of_Roesslers.py>`_); highlighted lines will be commented upon below:

.. literalinclude:: ../examples/SW_of_Roesslers.py
	:linenos:
	:dedent: 1
	:lines: 63-
	:emphasize-lines: 10, 26-27, 35, 32, 44

Explanation of selected features and choices:

* The values of :math:`ω` are initialised globally (line 10). We cannot just define a function here, because the parameter is used twice for each oscillator. Moreover, if we were trying to calculate Lyapunov exponents or the Jacobian, the generator function would be called multiple times, and thus the value of the parameter would not be consistent (which would be desastrous).

* Since we need :math:`\\sum_{j=0}^N x_j` to calculate the derivative of :math:`z` for every oscillator, it is prudent to only calculate this once. Therefore we define a helper symbol for this in lines 26 and 27, which we employ in line 35. (See the arguments of `jitcode` for details.)

* In line 32, we implement :math:`\\sin(t)`. For this purpose we had to import `t` in line 3. Also, we need to use `symengine.sin` – in contrast to `math.sin` or `numpy.sin`.

* As this is a large system, we use a generator function instead of a list to define :math:`f` (lines 29-36) and have the code automatically be split into chunks of 150 lines, corresponding to the equations of fifty oscillators, in line 44. (For this system, any reasonably sized multiple of 3 is a good choice for `chunk_size`; for other systems, the precise choice of the value may be more relevant.) See `large_systems` for the rationale.
"""

if __name__ == "__main__":
	def small_world_network(number_of_nodes, nearest_neighbours, rewiring_probability):
		n = number_of_nodes
		m = nearest_neighbours//2
		
		A = np.zeros( (n,n), dtype=bool )
		for i in range(n):
			for j in range(-m,m+1):
				A[i,(i+j)%n] = True
		
		# rewiring
		for i in range(n):
			for j in range(i):
				if A[i,j] and (rng.random() < rewiring_probability):
					A[j,i] = A[i,j] = False
					while True:
						i_new,j_new = rng.integers(0,n,2)
						if A[i_new,j_new] or i_new==j_new:
							continue
						else:
							A[j_new,i_new] = A[i_new,j_new] = True
							break
		
		return A
	
	# example-start
	import numpy as np
	import symengine

	from jitcode import jitcode, t, y
	
	# parameters
	# ----------
	
	rng = np.random.default_rng(seed=42)
	N = 500
	ω = rng.uniform(0.8,1.0,N)
	a = 0.165
	b = 0.2
	c = 10.0
	k = 0.01
	
	# get adjacency matrix of a small-world network
	A = small_world_network(
		number_of_nodes = N,
		nearest_neighbours = 20,
		rewiring_probability = 0.1
		)
	
	# generate differential equations
	# -------------------------------
	
	sum_z = symengine.Symbol("sum_z")
	helpers = [( sum_z, sum( y(3*j+2) for j in range(N) ) )]
	
	def f():
		for i in range(N):
			coupling_sum = sum( y(3*j)-y(3*i) for j in range(N) if A[i,j] )
			coupling_term = k * symengine.sin(t) * coupling_sum
			yield -ω[i] * y(3*i+1) - y(3*i+2) + coupling_term
			yield  ω[i] * y(3*i) + a*y(3*i+1)
			coupling_term_2 = k * (sum_z-N*y(3*i+2))
			yield b + y(3*i+2) * (y(3*i) - c) + coupling_term_2
	
	# integrate
	# ---------
	
	initial_state = rng.random(3*N)
	
	ODE = jitcode(f, helpers=helpers, n=3*N)
	ODE.generate_f_C(simplify=False, do_cse=False, chunk_size=150)
	ODE.set_integrator("dopri5")
	ODE.set_initial_value(initial_state,0.0)
	
	# data structure: x[0], v[0], z[0], x[1], …, x[N], v[N], z[N]
	data = np.vstack([ODE.integrate(T) for T in range(10,100000,10)])
