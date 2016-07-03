JiTCODE stands for just-in-time compilation for ordinary differential equations and is an extension of `SciPy’s ODE <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ (``scipy.integrate.ode``).
Where SciPy’s ODE takes a Python function as an argument, JiTCODE takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, which it translates to C code, compiles on the fly, and uses as the function to feed into SciPy’s ODE.

* `Documentation <http://jitcode.readthedocs.io>`_

* `Issue Tracker <http://github.com/neurophysik/jitcode/issues>`_

* Download from `PyPI <http://pypi.python.org/pypi/jitcode>`_ or just ``pip install jitcode``.