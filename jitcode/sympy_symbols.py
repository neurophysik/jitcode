import sympy

#: the symbol for the state that must be used to define the differential equation. It is a function and the integer argument denotes the component. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
y = sympy.Function("y",real=True)

#: the symbol for time for defining the differential equation. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
t = sympy.Symbol("t", real=True)
