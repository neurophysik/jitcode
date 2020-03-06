#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import sem as standard_error
from symengine import symbols

from jitcode import jitcode, y, jitcode_lyap, UnsuccessfulIntegration, test
from jitcode._jitcode import _is_C, _is_lambda

from scenarios import (
		y0, f_of_y0, jac_of_y0, y1, lyaps, vanilla, n,
	)

class TestOrders(unittest.TestCase):
	"""
	tests that the derivative and Jacobian are compiled/generated as intended when calling the several methods of the jitcode object in a certain order.
	"""
	def test_standard_order(self):
		self.ODE = jitcode(**vanilla)
		self.ODE.set_integrator("dopri5")
		self.ODE.set_initial_value(y0)
		self.assertTrue(_is_C(self.ODE.f))
		self.assertIsNone(self.ODE.jac)
	
	def test_standard_order_with_jac(self):
		self.ODE = jitcode(**vanilla)
		self.ODE.set_integrator("lsoda")
		self.ODE.set_initial_value(y0)
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def test_initialise_first(self):
		self.ODE = jitcode(**vanilla)
		self.ODE.set_initial_value(y0)
		self.ODE.set_integrator("dop853")
		self.assertTrue(_is_C(self.ODE.f))
	
	def test_initalise_with_dict(self):
		self.ODE = jitcode(**vanilla)
		initial_value = {y(i):y0[i] for i in range(n)}
		self.ODE.set_initial_value(initial_value)
		self.ODE.set_integrator("dop853")
		self.assertTrue(_is_C(self.ODE.f))
	
	def test_lambdas_with_jac(self):
		self.ODE = jitcode(wants_jacobian=True,**vanilla)
		self.ODE.generate_lambdas()
		self.ODE.set_integrator("lsoda")
		self.ODE.set_initial_value(y0)
		self.assertTrue(_is_lambda(self.ODE.f))
		self.assertTrue(_is_lambda(self.ODE.jac))
	
	def test_lambdas_without_jac(self):
		self.ODE = jitcode(**vanilla)
		self.ODE.generate_lambdas()
		self.assertIsNone(self.ODE.jac)
		self.ODE.set_integrator("vode")
		self.ODE.set_initial_value(y0)
	
	def test_compile_without_jac(self):
		self.ODE = jitcode(**vanilla)
		self.ODE.compile_C()
		self.ODE.set_integrator("LSODA")
		self.ODE.set_initial_value(y0)
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def test_generate_jac_manually(self):
		self.ODE = jitcode(**vanilla)
		self.ODE.generate_jac_C()
		self.ODE.set_integrator("vode")
		self.ODE.set_initial_value(y0)
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def test_save_and_load(self):
		self.ODE = jitcode(**vanilla)
		filename = self.ODE.save_compiled(overwrite=True)
		self.ODE = jitcode(n=n,module_location=filename)
		self.ODE.set_integrator("RK45")
		self.ODE.set_initial_value(y0)
		self.assertTrue(_is_C(self.ODE.f))
		self.assertIsNone(self.ODE.jac)
	
	def tearDown(self):
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.assertIsNotNone(self.ODE.f)
		assert_allclose( self.ODE.f(0.0,y0), f_of_y0, rtol=1e-5 )
		if not self.ODE.jac is None:
			assert_allclose( self.ODE.jac(0.0,y0), jac_of_y0, rtol=1e-5)
		assert_allclose( self.ODE.integrate(1.0), y1, rtol=1e-4 )
		for i in reversed(range(n)):
			assert_allclose( self.ODE.y_dict[y(i)], y1[i], rtol=1e-4 )

integrators = [
		#  name     with_jac
		("dopri5"  , False),
		("dop853"  , False),
		("lsoda"   , True ),
		("vode"    , True ),
		("RK23"    , False),
		("RK45"    , False),
		("Radau"   , True ),
		("BDF"     , True ),
		("LSODA"   , True ),
	]

class TestIntegrators(unittest.TestCase):
	"""
		Tests for every known integrator that its properties are identified correctly and the derivative and Jacobian are generated as needed.
	"""
	def test_normal_integration(self):
		for lambdas in (True,False):
			for integrator in integrators:
				with self.subTest(integrator=integrator,lambdas=lambdas):
					ODE = jitcode(**vanilla)
					if lambdas:
						ODE.generate_f_lambda()
						ODE.generate_jac_lambda()
					ODE.set_integrator(integrator[0])
					assert ODE._wants_jacobian == integrator[1]
					assert not  _is_C(ODE.f) == lambdas
					assert _is_lambda(ODE.f) == lambdas
					if integrator[1]:
						assert _is_lambda(ODE.jac) == lambdas
						assert not  _is_C(ODE.jac) == lambdas
					ODE.set_initial_value(y0,0.0)
					assert_allclose( ODE.integrate(1.0), y1, rtol=1e-3 )

class TestLyapunov(unittest.TestCase):
	"""
		Integration test for jitcode_lyap.
	"""
	def test_regular(self):
		self.ODE = jitcode_lyap(**vanilla,n_lyap=n)
		self.ODE.set_integrator("dopri5")
	
	def test_jac(self):
		self.ODE = jitcode_lyap(**vanilla,n_lyap=n)
		self.ODE.set_integrator("lsoda")
	
	def test_save_and_load_with_jac(self):
		self.ODE = jitcode_lyap(**vanilla,n_lyap=n,wants_jacobian=True)
		filename = self.ODE.save_compiled(overwrite=True)
		self.ODE = jitcode_lyap((),n=n,n_lyap=n,module_location=filename)
		self.ODE.set_integrator("vode")
		self.assertTrue(_is_C(self.ODE.f))
		self.assertTrue(_is_C(self.ODE.jac))
	
	def tearDown(self):
		if isinstance(self.ODE,jitcode) and self.ODE.f_sym():
			self.ODE.check()
		self.ODE.set_initial_value(y0,0.0)
		times = range(10,100000,10)
		data = np.vstack([ self.ODE.integrate(time)[1] for time in times ])
		result = np.average(data[1000:], axis=0)
		margin = standard_error(data[1000:], axis=0)
		self.assertLess( np.max(margin), 0.003 )
		for i in range(n):
			self.assertLess( result[i]-lyaps[i], 5*margin[i] )

class TestErrors(unittest.TestCase):
	def test_duplicate_error(self):
		ODE1 = jitcode(**vanilla)
		ODE2 = jitcode(**vanilla)
		ODE1.compile_C(modulename="foo")
		with self.assertRaises(NameError):
			ODE2.compile_C(modulename="foo")
	
	def test_wrong_n(self):
		with self.assertRaises(ValueError):
			ODE = jitcode(**vanilla,n=2*n)
	
	def test_dimension_mismatch(self):
		ODE = jitcode(**vanilla)
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
	
	def test_no_integrator(self):
		ODE = jitcode(**vanilla)
		with self.assertRaises(RuntimeError):
			ODE.integrate(1.0)
	
	def test_no_interpolation_LSODA(self):
		ODE = jitcode(**vanilla)
		with self.assertRaises(NotImplementedError):
			ODE.set_integrator("LSODA",interpolate=False)

if __name__ == "__main__":
	unittest.main(buffer=True)

