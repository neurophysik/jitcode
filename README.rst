JiTCODE stands for just-in-time compilation for ordinary differential equations and is an extension of `SciPy’s ODE`_ (`scipy.integrate.ode`).
Where SciPy’s ODE takes a Python function as an argument, JiTCODE takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ terms, which it translates to C code, compiles on the fly, and uses as the function to feed into SciPy’s ODE.

* `Documentation <http://jitcode.readthedocs.io>`_

* `Issue Tracker <http://github.com/neurophysik/jitcode/issues>`_