#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import platform
import shutil
import unittest
from tempfile import mkdtemp
from random import shuffle

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import sem as standard_error
from symengine import symbols

from jitcode import jitcode, y, jitcode_lyap, UnsuccessfulIntegration
from jitcode._jitcode import _is_C, _is_lambda

# control values:

# some state on the attractor
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

y1 = np.array([
	 0.0011789485114731,
	-0.0021947158873226,
	 0.0195744683782066,
	-0.0057801623466600,
	])

lyaps = [0.00710, 0, -0.0513, -0.187]

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

name = ""
def get_unique_name():
	global name
	name += "x"
	return name

class basic_test(unittest.TestCase):
	def tmpfile(self, filename):
		return os.path.join(self.directory, filename)
	
	@classmethod
	def setUpClass(self):
		self.argdict = {"f_sym": f}
		
	def setUp(self):
		self.directory = mkdtemp()
	
	def initialise_integrator(self):
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.ODE.set_initial_value(y0,0.0)
		self.extra_args = ()
	
	def test_standard_order(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.set_integrator('dopri5')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertIsNone(self.ODE.integrator.jac)
	
	def test_standard_order_with_jac(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.set_integrator('vode')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def test_heavily_chunked_f(self):
		self.ODE = jitcode(wants_jacobian=True, **self.argdict)
		self.ODE.generate_f_C(chunk_size=1,do_cse=True)
		self.ODE.set_integrator('dopri5')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
	
	def test_heavily_chunked_jac(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.generate_jac_C(chunk_size=1,do_cse=True)
		self.ODE.set_integrator('vode')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def test_heavily_chunked_helpers(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.generate_helpers_C(chunk_size=1)
		self.ODE.set_integrator('vode')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def test_omp(self):
		self.ODE = jitcode(wants_jacobian=True, **self.argdict)
		self.ODE.generate_f_C(chunk_size=1,do_cse=True)
		self.ODE.generate_jac_C(chunk_size=1,do_cse=True)
		self.ODE.generate_helpers_C(chunk_size=1)
		self.ODE.compile_C(omp=True)
		self.ODE.set_integrator('lsoda')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def test_initialise_first(self):
		self.ODE = jitcode(**self.argdict)
		self.initialise_integrator()
		self.ODE.set_integrator('dop853')
		self.assertTrue(_is_C(self.ODE.integrator.f))
	
	def test_lambdas(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.generate_lambdas()
		self.ODE.set_integrator('dopri5')
		self.initialise_integrator()
		self.assertTrue(_is_lambda(self.ODE.integrator.f))
		self.assertIsNone(self.ODE.integrator.jac)
	
	def test_lambdas_without_jac(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.generate_lambdas()
		self.assertIsNone(self.ODE.integrator.jac)
		self.ODE.set_integrator('vode')
		self.initialise_integrator()
	
	def test_compile_without_jac(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.compile_C()
		self.ODE.set_integrator('lsoda')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def test_generate_jac_manually(self):
		self.ODE = jitcode(**self.argdict)
		self.ODE.generate_jac_C()
		self.ODE.set_integrator('dopri5')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def test_save_and_load(self):
		self.ODE = jitcode(**self.argdict)
		destination = self.ODE.save_compiled(overwrite=True)
		folder, filename = os.path.split(destination)
		shutil.move(filename,self.tmpfile(filename))
		self.ODE = jitcode(n=len(f),module_location=self.tmpfile(filename))
		self.ODE.set_integrator('dopri5')
		self.initialise_integrator()
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertIsNone(self.ODE.integrator.jac)
	
	def tearDown(self):
		self.assertIsNotNone(self.ODE.integrator.f)
		assert_allclose( self.ODE.integrator.f(0.0,y0,*self.extra_args), f_of_y0, rtol=1e-5 )
		if not self.ODE.integrator.jac is None:
			assert_allclose( self.ODE.integrator.jac(0.0,y0,*self.extra_args), jac_of_y0, rtol=1e-5)
		assert_allclose( self.ODE.integrate(1.0), y1, rtol=1e-5 )
		if platform.system() != "Windows":
			# Windows blocks loaded module files from removal.
			shutil.rmtree(self.directory)

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

class basic_test_with_helpers(basic_test):
	@classmethod
	def setUpClass(self):
		self.argdict = {"f_sym": f_alt, "helpers": get_f_alt_helpers()}

class helpers_test(unittest.TestCase):
	def test_identity_of_jacs(self):
		x = np.random.random(len(f))
		ODE1 = jitcode(f)
		ODE2 = jitcode(f_alt, get_f_alt_helpers())
		ODE1.set_integrator("vode")
		ODE2.set_integrator("vode")
		assert_allclose(ODE1.integrator.jac(0.0,x), ODE2.integrator.jac(0.0,x))
	
	def test_identity_of_lyaps(self):
		n = len(f)
		x = np.random.random((n+1)*n)
		ODE1 = jitcode_lyap(f,n_lyap=n)
		ODE2 = jitcode_lyap(f_alt,helpers=get_f_alt_helpers(),n_lyap=n)
		ODE1.set_integrator("dopri5")
		ODE2.set_integrator("dopri5")
		assert_allclose(ODE1.integrator.f(0.0,x), ODE2.integrator.f(0.0,x))

class lyapunov_test(unittest.TestCase):
	def setUp(self):
		self.n = len(f)
	
	def test_lyapunov(self):
		self.ODE = jitcode_lyap(f,n_lyap=self.n)
		self.ODE.set_integrator("dopri5")
	
	def test_lyapunov_with_jac(self):
		self.ODE = jitcode_lyap(f,n_lyap=self.n)
		self.ODE.set_integrator("vode")
	
	def test_lyapunov_with_helpers(self):
		self.ODE = jitcode_lyap(f_alt,helpers=get_f_alt_helpers(),n_lyap=self.n)
		self.ODE.set_integrator("dopri5")
	
	def test_lyapunov_with_helpers_and_jac(self):
		self.ODE = jitcode_lyap(f_alt,helpers=get_f_alt_helpers(),n_lyap=self.n)
		self.ODE.set_integrator("vode")
	
	def test_lyapunov_save_and_load_with_jac(self):
		self.ODE = jitcode_lyap(f,n_lyap=self.n,wants_jacobian=True)
		filename = self.ODE.save_compiled(overwrite=True)
		self.ODE = jitcode_lyap((),n=self.n,n_lyap=self.n,module_location=filename)
		self.ODE.set_integrator("vode")
		self.assertTrue(_is_C(self.ODE.integrator.f))
		self.assertTrue(_is_C(self.ODE.integrator.jac))
	
	def initialise_integrator(self):
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.ODE.set_initial_value(y0,0.0)
	
	def tearDown(self):
		self.initialise_integrator()
		times = range(10,100000,10)
		data = np.vstack( self.ODE.integrate(time)[1] for time in times )
		result = np.average(data[1000:], axis=0)
		margin = standard_error(data[1000:], axis=0)
		print(data,result,margin)
		self.assertLess( np.max(margin), 0.003 )
		for i in range(self.n):
			self.assertLess( result[i]-lyaps[i], 3*margin[i] )

def f_generator():
	for entry in f:
		yield entry

class basic_test_with_generator_function(basic_test):
	@classmethod
	def setUpClass(self):
		self.argdict = {"f_sym": f_generator, "n": len(f)}

a_par, c_par, k_par = symbols("a_par c_par k_par")
f_params_helpers = [ ( coupling, k_par*(y(2)-y(0)) ) ]
f_params = [
	y(0) * ( a_par-y(0) ) * ( y(0)-1.0 ) - y(1) + coupling,
	b1*y(0) - c_par*y(1),
	y(2) * ( a_par-y(2) ) * ( y(2)-1.0 ) - y(3) - coupling,
	b2*y(2) - c_par*y(3)
	]
params_args = (a,c,k)

class basic_test_with_params(basic_test):
	@classmethod
	def setUpClass(self):
		self.argdict = {
				"f_sym": f_params,
				"helpers": f_params_helpers,
				"control_pars": [a_par, c_par, k_par]
			}
	
	def initialise_integrator(self):
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.ODE.set_initial_value(y0,0.0)
		self.ODE.set_f_params(*params_args)
		self.ODE.set_jac_params(*params_args)
		self.extra_args = params_args

class errors_test(unittest.TestCase):
	def test_duplicate_error(self):
		ODE1 = jitcode(f)
		ODE2 = jitcode(f)
		ODE1.compile_C(modulename="foo")
		with self.assertRaises(NameError):
			ODE2.compile_C(modulename="foo")
	
	def test_wrong_n(self):
		with self.assertRaises(ValueError):
			ODE = jitcode(f,n=len(f)*2)
	
	def test_dimension_mismatch(self):
		ODE = jitcode(f)
		with self.assertRaises(ValueError):
			ODE.set_initial_value(np.array([1.,2.,3.]),0.0)
	
	def test_check_index_negative(self):
		ODE = jitcode([y(-1)])
		with self.assertRaises(ValueError):
			ODE.check()
	
	def test_check_index_too_high(self):
		ODE = jitcode([y(1)])
		with self.assertRaises(ValueError):
			ODE.check()
	
	def test_check_undefined_variable(self):
		x = symbols("x")
		ODE = jitcode([x])
		with self.assertRaises(ValueError):
			ODE.check()
	
	def test_backwards_integration(self):
		ODE = jitcode(f)
		ODE.set_initial_value([0,1,2,3],0)
		ODE.set_integrator("dopri5")
		with self.assertRaises(ValueError):
			ODE.integrate(-1)
	
	def test_zero_integration(self):
		ODE = jitcode(f)
		initial = np.random.random(4)
		ODE.set_initial_value(initial,0)
		ODE.set_integrator("dopri5")
		assert_allclose(initial,ODE.integrate(0))
	
	def test_failed_integration(self):
		ODE = jitcode(f)
		ODE.set_initial_value([0,1,2,3],0)
		ODE.set_integrator("dopri5",atol=1e-10,rtol=0,nsteps=10)
		with self.assertRaises(UnsuccessfulIntegration):
			ODE.integrate(100)

if __name__ == "__main__":
	unittest.main(buffer=True)

