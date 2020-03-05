from numpy import pi, vstack, savetxt
from jitcode import jitcode, y
from symengine import sin
from numpy.random import choice, uniform

n = 100
c = 3.0
q = 0.2

A = choice( [1,0], size=(n,n), p=[q,1-q] )

omega = uniform(-0.5,0.5,n)
# Sorting does not effect the qualitative dynamics, but only the resulting plots:
omega.sort()

def kuramotos_f():
	for i in range(n):
		coupling_sum = sum(
				sin(y(j)-y(i))
				for j in range(n)
				if A[j,i]
			)
		yield omega[i] + c/(n-1)*coupling_sum

I = jitcode(kuramotos_f,n=n)
I.set_integrator("dop853",atol=1e-6,rtol=0)

initial_state = uniform(0,2*pi,n)
I.set_initial_value(initial_state,time=0.0)

times = range(0,2001)
data = vstack ([ I.integrate(time) for time in times ])
data %= 2*pi
savetxt("kuramotos.dat",data)

