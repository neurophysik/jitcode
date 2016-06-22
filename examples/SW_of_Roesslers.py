#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This more complicated example showcases several advanced features of JiTCODE that are relevant for an efficient integration of more complex systems.

Suppose we want to integrate a system of $N=600$ Rössler oscillators, with the $i$th oscillator being described by the following differential equations (note that we used $v$ instead of $y$ for the third dynamical variable to avoid an overloading of symbols):

.. math::
	\\begin{alignedat}{1}
	\\dot{x}_i = -ω_i v_i &- z_i + k \\sum_{j=0}^N A_{ij} (x_j-x_i) \\
	\\dot{v}_i = ω_i x_i &+ a v_i
	\\end{alignedat}\\
	\\dot{z}_i = b + z_i (x_i -c) + k \\sum_{j=0}^N (x_j-x_i)

The control parameters shall be $a = 0.165$, $b = 0.2$, $c = 10.0$, and $k = 0.01$. The (frequency) parameter $ω$ shall be picked randomly from the uniform distribution on $[0.8,1.0]$. $A∈ℝ^{N×N}$ shall be the adjacency matrix of a one-dimensional small-world network (which shall be provided by a function `small_world_network` in the example code).
ü"""

if __name__ == "__main__":
	def small_world_network(n, m, p):
		A = np.zeros( (n,n), dtype=bool )
		for i in range(n):
			for j in range(-m,m+1):
				A[i,(i+j)%n] = True
		
		# rewiring
		for i in range(n):
			for j in range(j):
				if A[i,j] and (np.random.random() < p):
					A[j,i] = A[i,j] = False
					while True:
						i_new,j_new = np.random.randint(0,n,2)
						if A[i_new,j_new] or i_new==j_new:
							continue
						else:
							A[j_new,i_new] = A[i_new,j_new] = True
							break
		
		return A
	
	# example-start
	from jitcode import jitcode, provide_basic_symbols
	import numpy as np
	import sympy
	
	# generate adjacency matrix of one-dimensional small-world
	# --------------------------------------------------------
	
	# network parameters
	N = 600	# number of nodes
	m = 10	# number of nearest neighbours on each side
	p = 0.1	# rewiring probability
	
	# adjacency matrix of a small-world network
	A = small_world_network(N, m, p)
	
	# generate differential equations
	# -------------------------------
	
	t, y = provide_basic_symbols()
	
	# control parameters
	ω = np.random.uniform(0.8,1.0,N)
	a = 0.165
	b = 0.2
	c = 10.0
	k = 0.01
	
	mean_z = sympy.Symbol("mean_z")
	j = sympy.Symbol("j")
	helpers = [( mean_z, sympy.Sum( y(3*j+2), (j,0,N-1) ) / N )]
	
	def f():
		for i in range(N):
			coupling_term = sympy.Mul(
				k,
				sum( (y(3*j)-y(3*i)) for j in range(N) if A[i,j] ),
				evaluate = False
			)
			yield -ω[i] * y(3*i+1) - y(3*i+2) + coupling_term
			yield  ω[i] * y(3*i) + a*y(3*i+1)
			yield b + y(3*i+2) * (y(3*i) - c) + k * (y(3*i+2)-mean_z)
	
	# integrate
	# ---------
	
	initial_state = np.random.random(3*N)
	
	ODE = jitcode(f, helpers=helpers, n=3*N)
	ODE.generate_f_C(simplify=False, do_cse=False, chunk_size=120)
	ODE.set_integrator('dopri5')
	ODE.set_initial_value(initial_state,0.0)
	
	# data structure: x[0], …, x[N], v[0], …, v[N], z[0], …, z[N]
	data = np.vstack(ODE.integrate(t) for t in range(10,100000,10))
