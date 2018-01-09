#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import platform
import shutil
import unittest
from tempfile import mkdtemp
from random import shuffle, sample

import numpy as np
from numpy.testing import assert_allclose
from symengine import symbols

from jitcode import jitcode, y, jitcode_lyap
from jitcode._jitcode import _is_C, _is_lambda

# control values:

y0 = np.array([ -0.00338158, -0.00223185, 0.01524253, -0.00613449 ])

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

class TestBasic(unittest.TestCase):
	params = ()
	
	def setUp(self):
		self.ODE = jitcode(f_sym=f)
	
	def test_default(self):
		self.ODE.generate_f_C()
		self.ODE.generate_jac_C()
		self.ODE.compile_C()
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def test_heavily_chunked(self):
		self.ODE.generate_f_C(chunk_size=1,do_cse=True)
		self.ODE.generate_jac_C(chunk_size=1,do_cse=True)
		self.ODE.generate_helpers_C(chunk_size=1)
		self.ODE.compile_C()
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def test_omp(self):
		self.ODE.generate_f_C(chunk_size=1,do_cse=True)
		self.ODE.generate_jac_C(chunk_size=1,do_cse=True)
		self.ODE.generate_helpers_C(chunk_size=1)
		self.ODE.compile_C(omp=True)
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def test_lambdas_with_jac(self):
		self.ODE.generate_f_lambda()
		self.ODE.generate_jac_lambda()
		self.assertTrue(_is_lambda(self.ODE.f))
		self.assertTrue(_is_lambda(self.ODE.jac))
	
	def tearDown(self):
		self.assertIsNotNone(self.ODE.f)
		assert_allclose( self.ODE.f(0.0,y0,*self.params), f_of_y0, rtol=1e-5 )
		if not self.ODE.jac is None:
			assert_allclose( self.ODE.jac(0.0,y0,*self.params), jac_of_y0, rtol=1e-5)

#----------------------------

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

def get_f_alt_helpers():
	shuffle(f_alt_helpers)
	return f_alt_helpers

class TestHelpers(TestBasic):
	def setUp(self):
		self.ODE = jitcode(f_alt,helpers=get_f_alt_helpers())

#----------------------------

class helpers_test(unittest.TestCase):
	def test_identity_of_jacs(self):
		ODE1 = jitcode(f)
		ODE1.generate_jac_C()
		ODE1.compile_C()
		
		ODE2 = jitcode(f_alt, get_f_alt_helpers())
		ODE2.generate_jac_C()
		ODE2.compile_C()
		
		x = np.random.random(len(f))
		assert_allclose(ODE1.jac(0.0,x), ODE2.jac(0.0,x))
	
	def test_identity_of_lyaps(self):
		n = len(f)
		x = np.random.random((n+1)*n)
		ODE1 = jitcode_lyap(f,n_lyap=n)
		ODE2 = jitcode_lyap(f_alt,helpers=get_f_alt_helpers(),n_lyap=n)
		ODE1.compile_C()
		ODE2.compile_C()
		assert_allclose(ODE1.f(0.0,x), ODE2.f(0.0,x))

# -----------------

def f_generator():
	for entry in f:
		yield entry

class TestGenerator(TestBasic):
	def setUp(self):
		self.ODE = jitcode(f_generator, n=len(f))

# -------------------

a_par, c_par, k_par = symbols("a_par c_par k_par")
f_params_helpers = [ ( coupling, k_par*(y(2)-y(0)) ) ]
f_params = [
	y(0) * ( a_par-y(0) ) * ( y(0)-1.0 ) - y(1) + coupling,
	b1*y(0) - c_par*y(1),
	y(2) * ( a_par-y(2) ) * ( y(2)-1.0 ) - y(3) - coupling,
	b2*y(2) - c_par*y(3)
	]

class TestParams(TestBasic):
	params = (a,c,k)
	
	def setUp(self):
		self.ODE = jitcode(
				f_sym = f_params,
				helpers = f_params_helpers,
				control_pars = [a_par, c_par, k_par]
			)
	
	def initialise_integrator(self):
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.ODE.set_parameters(*params_args)
		self.ODE.set_initial_value(y0,0.0)
		self.extra_args = params_args

if __name__ == "__main__":
	unittest.main(buffer=True)

