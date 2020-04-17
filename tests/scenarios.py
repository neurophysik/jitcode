#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from symengine import symbols, Function
from jitcode import y
from random import shuffle

n = 4

# initial condition
y0 = np.array([ -0.00338158, -0.00223185, 0.01524253, -0.00613449 ])

# solutions
f_of_y0 = np.array([
		0.0045396904008868,
		0.00002265673,
		0.0043665702488807,
		0.000328463955
	])

jac_of_y0 = np.array([
		[-0.1088290163008492, -1.  ,  0.128             ,  0.   ],
		[ 0.0065            , -0.02,  0.                ,  0.   ],
		[ 0.128             ,  0.  , -0.0732042758000427, -1.   ],
		[ 0.                ,  0.  ,  0.0135            , -0.02 ]
	])

y1 = np.array([
		 0.0011789485114731,
		-0.0021947158873226,
		 0.0195744683782066,
		-0.0057801623466600,
	])

lyaps = [0.00710, 0, -0.0513, -0.187]

# vanilla

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

vanilla = {"f_sym": f}

# as dictionary
f_dict = { y(i):entry for i,entry in reversed(list(enumerate(f))) }
with_dictionary = {"f_sym": f_dict}

# with generator

def f_generator():
	for entry in f:
		yield entry

with_generator = { "f_sym":f_generator, "n":n }

# with helpers

f1, f2, f3, f4 = symbols("f1, f2, f3, f4")
coupling, first_y, first_y_sq = symbols("coupling, first_y, first_y_sq")
a_alt, b1_alt, b2_alt, c_alt, k_alt = symbols("a_alt, b1_alt, b2_alt, c_alt, k_alt")
f_alt = [ f1, f2, f3, f4 ]

f_alt_helpers = [
	( a_alt , a  ),
	( b1_alt, b1 ),
	( b2_alt, b2 ),
	( c_alt,  c  ),
	( k_alt,  k  ),
	( first_y, y(0) ),
	( first_y_sq, first_y**2 ),
	( coupling, k_alt*(y(2)-first_y) ),
	( f1, a_alt*first_y_sq - a_alt*first_y + coupling - first_y**3 + first_y_sq - y(1) ),
	( f2, b1_alt*first_y - c_alt*y(1)),
	( f3, y(2) * ( a_alt-y(2) ) * ( y(2)-1.0 ) - y(3) - coupling ),
	( f4, b2_alt*y(2) - c_alt*y(3) ),
]
shuffle(f_alt_helpers)

with_helpers = {"f_sym": f_alt, "helpers": f_alt_helpers}

# with parameters

a_par, c_par, k_par = symbols("a_par c_par k_par")
f_params_helpers = [ ( coupling, k_par*(y(2)-y(0)) ) ]
f_params = [
	y(0) * ( a_par-y(0) ) * ( y(0)-1.0 ) - y(1) + coupling,
	b1*y(0) - c_par*y(1),
	y(2) * ( a_par-y(2) ) * ( y(2)-1.0 ) - y(3) - coupling,
	b2*y(2) - c_par*y(3)
	]
params_args = (a,c,k)
n_params = 3

with_params = {
		"f_sym": f_params,
		"helpers": f_params_helpers,
		"control_pars": [a_par, c_par, k_par]
	}

# with callbacks

def first_component(Y):
	return Y[0] * ( a-Y[0] ) * ( Y[0]-1.0 ) - Y[1] + k * (Y[2] - Y[0])

c_times = lambda y,arg: c*arg

call_first_component = Function("call_first_component")
call_c_times = Function("call_c_times")

f_callback = [
		call_first_component(),
		b1*y(0) - call_c_times(y(1)),
		y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
		b2*y(2) - c*y(3),
	]

callback = {
		"f_sym": f_callback,
		"callback_functions": [
			(call_first_component,first_component,0),
			(call_c_times,c_times,1)
		],
	}


