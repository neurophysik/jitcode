#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.random import random

from jitcode import jitcode
from jitcode.integrator_tools import empty_integrator, IVP_wrapper, IVP_wrapper_no_interpolation, ODE_wrapper, UnsuccessfulIntegration

from scenarios import y0, y1, vanilla, with_params, params_args, n, n_params

ODE = jitcode(**vanilla,verbose=False,wants_jacobian=True)
ODE.compile_C()
f,jac = ODE.f,ODE.jac

# … time-inverted for certain failure

ODE = jitcode(
		f_sym = [-entry for entry in vanilla["f_sym"]],
		verbose=False,
		wants_jacobian=True
	)
ODE.compile_C()
f_back,jac_back = ODE.f,ODE.jac

# … and once more with parameters

ODE = jitcode(
		**with_params,
		wants_jacobian = True,
		verbose = False,
	)
ODE.compile_C()
f_params,jac_params = ODE.f,ODE.jac

# -----------------------------

class TestSkeleton(object):
	def control_result(self):
		result = self.integrator.integrate(1.0)
		assert_allclose( result, y1, rtol=1e-3 )
	
	def test_no_params(self):
		self.initialise(f,jac,rtol=1e-5)
		self.integrator.set_initial_value(y0)
		self.control_result()
	
	def test_initial_twice(self):
		self.initialise(f,jac,rtol=1e-5)
		self.integrator.set_initial_value(random(n))
		self.integrator.set_initial_value(y0)
		self.control_result()
	
	def test_params(self):
		self.initialise(f_params,jac_params,rtol=1e-5)
		self.integrator.set_params(*params_args)
		self.integrator.set_initial_value(y0)
		self.control_result()
	
	def test_params_other_order(self):
		self.initialise(f_params,jac_params,rtol=1e-5)
		self.integrator.set_initial_value(y0)
		self.integrator.set_params(*params_args)
		self.control_result()
	
	def test_params_twice(self):
		self.initialise(f_params,jac_params,rtol=1e-5)
		self.integrator.set_params(*random(n_params))
		self.integrator.set_initial_value(y0)
		self.integrator.set_params(*params_args)
		self.control_result()
	
	def test_zero_integration(self):
		self.initialise(f,jac)
		initial = random(n)
		self.integrator.set_initial_value(initial)
		assert_allclose(initial,self.integrator.integrate(0))
	
	def test_backwards_integration(self):
		self.initialise(f,jac)
		self.integrator.set_initial_value([0,1,2,3],0)
		with self.assertRaises(ValueError):
			self.integrator.integrate(-1)
	
	def test_failed_integration(self):
		self.initialise(f_back,jac_back,atol=1e-12,rtol=1e-12)
		self.integrator.set_initial_value([0,1,2,3],0)
		with self.assertRaises(UnsuccessfulIntegration):
			self.integrator.integrate(1)

class TestRK45(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper("RK45",f,**kwargs)

class TestRK23(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper("RK45",f,**kwargs)

class TestRadau(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper("Radau",f,jac,**kwargs)

class TestBDF(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper("BDF",f,jac,**kwargs)

class TestRK45_no_interpolation(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper_no_interpolation("RK45",f,**kwargs)

class TestRK23_no_interpolation(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper_no_interpolation("RK45",f,**kwargs)

class TestRadau_no_interpolation(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper_no_interpolation("Radau",f,jac,**kwargs)

class TestBDF_no_interpolation(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper_no_interpolation("BDF",f,jac,**kwargs)

class TestLSODA(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = IVP_wrapper("LSODA",f,jac,**kwargs)
	
	def test_failed_integration(self):
		pass

class TestDopri5(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = ODE_wrapper(f)
		self.integrator.set_integrator("dopri5")

class TestDop853(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = ODE_wrapper(f)
		self.integrator.set_integrator("dop853")

class TestLsoda(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = ODE_wrapper(f,jac)
		self.integrator.set_integrator("lsoda")
	
	def test_failed_integration(self):
		pass

class TestVode(unittest.TestCase,TestSkeleton):
	def initialise(self,f,jac,**kwargs):
		self.integrator = ODE_wrapper(f,jac)
		self.integrator.set_integrator("vode")
	
	def test_failed_integration(self):
		pass

class TestDummy(unittest.TestCase):
	def setUp(self):
		self.integrator = empty_integrator()
	
	def test_t(self):
		with self.assertRaises(RuntimeError):
			self.integrator.t
	
	def test_set_integrator(self):
		with self.assertRaises(RuntimeError):
			self.integrator.set_integrator("")
	
	def test_integrate(self):
		with self.assertRaises(RuntimeError):
			self.integrator.integrate(2.3)
	
	def test_set_initial(self):
		initial = random(5)
		self.integrator.set_initial_value(initial,1.2)
		assert np.all( self.integrator._y == initial )
		assert self.integrator.t == 1.2
	
	def test_set_parameters(self):
		params = random(5)
		self.integrator.set_params(params)
		assert np.all( self.integrator.params == params )

if __name__ == "__main__":
	unittest.main(buffer=True)

