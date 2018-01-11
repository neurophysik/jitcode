#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Tests that the code-generation and compilaton or lambdification, respectively, of the derivative and Jacobian works as intended in all kinds of scenarios.
"""

import unittest

from numpy.random import random
from numpy.testing import assert_allclose

from jitcode import jitcode, jitcode_lyap
from jitcode._jitcode import _is_C, _is_lambda

from scenarios import (
		y0, f_of_y0, jac_of_y0,
		vanilla, with_params, with_helpers, with_generator,
		n, params_args
	)

class TestBasic(unittest.TestCase):
	params = ()
	
	def setUp(self):
		self.ODE = jitcode(**vanilla)
	
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

class TestHelpers(TestBasic):
	def setUp(self):
		self.ODE = jitcode(**with_helpers)

class FurtherHelpersTests(unittest.TestCase):
	def test_identity_of_jacs(self):
		x = random(n)
		
		def evaluate(scenario):
			ODE = jitcode(**scenario)
			ODE.generate_jac_C()
			ODE.compile_C()
			return ODE.jac(0.0,x)
		
		ODE2 = jitcode(**with_helpers)
		ODE2.generate_jac_C()
		ODE2.compile_C()
		
		assert_allclose(
				evaluate(vanilla),
				evaluate(with_helpers)
			)
	
	def test_identity_of_lyaps(self):
		x = random((n+1)*n)
		
		def evaluate(scenario):
			ODE = jitcode_lyap(**scenario,n_lyap=n)
			ODE.compile_C()
			return ODE.f(0.0,x)
		
		assert_allclose(
				evaluate(vanilla),
				evaluate(with_helpers)
			)

class TestGenerator(TestBasic):
	def setUp(self):
		self.ODE = jitcode(**with_generator)

class TestParams(TestBasic):
	params = params_args
	
	def setUp(self):
		self.ODE = jitcode(**with_params)

if __name__ == "__main__":
	unittest.main(buffer=True)

