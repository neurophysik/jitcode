JiTCODE stands for just-in-time compilation for ordinary differential equations and is an extension of `SciPy’s ODE <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ (``scipy.integrate.ode``).
Where SciPy’s ODE takes a Python function as an argument, JiTCODE takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, which it translates to C code, compiles on the fly, and uses as the function to feed into SciPy’s ODE.
If you want to integrate delay or stochastic differential equations, check out
`JiTCDDE <http://github.com/neurophysik/jitcdde>`_, or
`JiTCSDE <http://github.com/neurophysik/jitcsde>`_, respectively.


* `Documentation <http://jitcode.readthedocs.io>`_ – Read this to get started and for reference. Don’t miss that some topics are addressed in the `common JiTC*DE documentation <http://jitcde-common.readthedocs.io>`_.

* `Paper <https://doi.org/10.1063/1.5019320>`_ – Read this for the scientific background. Cite this (`BibTeX <https://raw.githubusercontent.com/neurophysik/jitcxde_common/master/citeme.bib>`_) if you wish to give credit or to shift blame.

* `Issue Tracker <http://github.com/neurophysik/jitcode/issues>`_ – Please report any bugs here. Also feel free to ask for new features.

* `Installation instructions <http://jitcde-common.readthedocs.io/#installation>`_. In most cases, `pip3 install jitcode` or similar should do the job.

This work was supported by the Volkswagen Foundation (Grant No. 88463).

