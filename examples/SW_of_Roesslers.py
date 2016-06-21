#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
TODO: Explanation
"""

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
	
	A = small_world_network(N, m, p)
	
	# generate differential equations
	# -------------------------------
	
	t, y = provide_basic_symbols()
	
	# control parameters
	ω = np.random.uniform(0.8,1.0,N)
	a = 0.165
	b = 0.2
	c = 10.0
	coupling_strength = 0.01
	
	mean_z = sympy.Symbol("mean_z")
	k = sympy.Symbol("k")
	
	helpers = [( mean_z, sympy.Sum( y(3*k+2), (k,0,N-1) ) / N )]
	
	def f():
		for i in range(N):
			coupling_term = sympy.Mul(
				coupling_strength,
				sum( (y(3*j)-y(3*i)) for j in range(N) if A[i,j] ),
				evaluate = False
			)
			yield -ω[i] * y(3*i+1) - y(3*i+2) + coupling_term
			yield  ω[i] * y(3*i) + a*y(3*i+1)
			yield b + y(3*i+2) * (y(3*i) - c) + coupling_strength * (y(3*i+2)-mean_z)
	
	# integrate
	# ---------
	
	initial_state = np.random.random(3*N)
	
	ODE = jitcode(f, helpers=helpers, n=3*N)
	ODE.generate_f_C(simplify=False, do_cse=False, chunk_size=120)
	ODE.set_integrator('dopri5')
	ODE.set_initial_value(initial_state,0.0)
	
	data = np.vstack(ODE.integrate(t) for t in range(10,1000,10))
