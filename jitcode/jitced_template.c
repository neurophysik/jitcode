# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# include <Python.h>
# include <numpy/arrayobject.h>
# include <math.h>

# define TYPE_INDEX NPY_DOUBLE

unsigned int const dimension={{n}};

{% for control_par in control_pars %}
double parameter_{{control_par}};
{% endfor %}

{% if callbacks|length %}
static inline double callback(PyObject * Python_function, PyObject * arglist)
{
	PyObject * py_result = PyObject_CallObject(Python_function,arglist);
	Py_DECREF(arglist);
	double result = PyFloat_AsDouble(py_result);
	Py_DECREF(py_result);
	return result;
}
{% endif %}

{% for function,nargs in callbacks %}
static PyObject * callback_{{function}};
# define {{function}}(...) callback(\
		callback_{{function}}, \
		Py_BuildValue( \
				{% if nargs -%}
				"(O{{'d'*nargs}})", Y, __VA_ARGS__ \
				{% else -%}
				"(O)", Y \
				{% endif -%}
			))
{% endfor %}

# define get_general_helper(i) ((general_helper[i]))
# define set_general_helper(i,value) (general_helper[i] = value)

# define get_f_helper(i) ((f_helper[i]))
# define set_f_helper(i,value) (f_helper[i] = value)

{% if has_Jacobian: %}
# define get_jac_helper(i) ((jac_helper[i]))
# define set_jac_helper(i,value) (jac_helper[i] = value)
{% endif %}

# define y(i) (* (double *) PyArray_GETPTR1(Y, i))

# define set_dy(i, value) (* (double *) PyArray_GETPTR1(dY, i) = value)

{% if has_Jacobian: %}
#define set_dfdy(i, j, value) (* (double *) PyArray_GETPTR2(dfdY, i, j) = value)
{% endif %}

{% if number_of_general_helpers>0: %}
# include "general_helpers_definitions.c"
{% endif %}

{% if number_of_f_helpers>0: %}
# include "f_helpers_definitions.c"
{% endif %}

# include "f_definitions.c"

static PyObject * py_f(PyObject *self, PyObject *args)
{
	double t;
	PyArrayObject * Y;
	
	if (!PyArg_ParseTuple(
				args,
				"dO!",
				&t,
				&PyArray_Type, &Y
			))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	if (PyArray_NDIM(Y) != 1)
	{
		PyErr_SetString(PyExc_ValueError,"Array must be one-dimensional.");
		return NULL;
	}
	else if ((PyArray_TYPE(Y) != TYPE_INDEX))
	{
		PyErr_SetString(PyExc_TypeError,"Array needs to be of type double.");
		return NULL;
	}
	
	npy_intp dims[1] = {dimension};
	PyArrayObject * dY = (PyArrayObject *) PyArray_EMPTY(1, dims, TYPE_INDEX, 0);
	
	if (dY == NULL)
	{
		PyErr_SetString (PyExc_ValueError, "Error: Could not allocate array.");
		exit(1);
	}
	
	{% if number_of_general_helpers>0: %}
	double general_helper[{{number_of_general_helpers}}];
	# include "general_helpers.c"
	{% endif %}
	
	{% if number_of_f_helpers>0: %}
	double f_helper[{{number_of_f_helpers}}];
	# include "f_helpers.c"
	{% endif %}
	
	# include "f.c"
	
	return PyArray_Return(dY);
}

{% if has_Jacobian: %}
{% if number_of_jac_helpers>0: %}
# include "jac_helpers_definitions.c"
{% endif %}
# include "jac_definitions.c"

static PyObject * py_jac(PyObject *self, PyObject *args)
{
	double t;
	PyArrayObject * Y;
	
	if (!PyArg_ParseTuple(
				args,
				"dO!",
				&t,
				&PyArray_Type, &Y
			))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	if (PyArray_NDIM(Y) != 1)
	{
		PyErr_SetString(PyExc_ValueError,"Array must be one-dimensional.");
		return NULL;
	}
	else if ((PyArray_TYPE(Y) != TYPE_INDEX))
	{
		PyErr_SetString(PyExc_TypeError,"Array needs to be of type double.");
		return NULL;
	}
	
	npy_intp dims[2] = {dimension, dimension};
	
	{% if sparse_jac: %}
	PyArrayObject * dfdY = (PyArrayObject *) PyArray_ZEROS(2, dims, TYPE_INDEX, 0);
	{% else: %}
	PyArrayObject * dfdY = (PyArrayObject *) PyArray_EMPTY(2, dims, TYPE_INDEX, 0);
	{% endif %}
	
	if (dfdY == NULL)
	{
		PyErr_SetString (PyExc_ValueError, "Error: Could not allocate array.");
		exit(1);
	}
	
	{% if number_of_general_helpers>0: %}
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-Wunused-but-set-variable"
	double general_helper[{{number_of_general_helpers}}];
	# include "general_helpers.c"
	# pragma GCC diagnostic pop
	{% endif %}
	
	{% if number_of_jac_helpers>0: %}
	double jac_helper[{{number_of_jac_helpers}}];
	# include "jac_helpers.c"
	{% endif %}
	
	# include "jac.c"
	
	return PyArray_Return(dfdY);
}
{% endif %}

static PyObject * py_initialise(PyObject *self, PyObject * args)
{
	if (!PyArg_ParseTuple(
		args,
		"{{'d'*control_pars|length}}{{'O'*callbacks|length}}"
		{% for control_par in control_pars %}
		, &parameter_{{control_par}}
		{% endfor %}
		{% for function,nargs in callbacks %}
		, &callback_{{function}}
		{% endfor %}
		))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	{% for function,nargs in callbacks %}
	if (!PyCallable_Check(callback_{{function}}))
	{
		PyErr_SetString(PyExc_TypeError,"Callback must be callable.");
		return NULL;
	}
	{% endfor %}
	
	Py_RETURN_NONE;
}

#define SIGNATURE "(t,y)\n--\n\n"

static PyMethodDef {{module_name}}_methods[] = {
	{"initialise", (PyCFunction) py_initialise, METH_VARARGS, NULL},
	{"f", py_f, METH_VARARGS, "f" SIGNATURE},
	{% if has_Jacobian: %}
	{"jac", py_jac, METH_VARARGS, "jac" SIGNATURE},
	{% endif %}
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef =
{
	PyModuleDef_HEAD_INIT,
	"{{module_name}}",
	NULL,
	-1,
	{{module_name}}_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_{{module_name}}(void)
{
    PyObject * module = PyModule_Create(&moduledef);
	import_array();
    return module;
}

