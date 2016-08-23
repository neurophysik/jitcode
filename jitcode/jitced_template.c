# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-pedantic"
# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# include <Python.h>
# include <numpy/arrayobject.h>
# pragma GCC diagnostic pop

# include <math.h>

# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"

# define TYPE_INDEX NPY_DOUBLE

unsigned int const dimension={{n}};

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
static void general(PyArrayObject *restrict const Y, double *restrict const general_helper)
{
	# include "general_helpers.c"
}
{% endif %}

{% if number_of_f_helpers>0: %}
# include "f_helpers_definitions.c"
{% endif %}
# include "f_definitions.c"

static PyObject * py_f(PyObject *self, PyObject *args)
{
	double t;
	PyArrayObject * Y;
	
	if (!PyArg_ParseTuple(args, "dO!", &t, &PyArray_Type, &Y))
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
	general(Y, general_helper);
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
	
	if (!PyArg_ParseTuple(args, "dO!", &t, &PyArray_Type, &Y))
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
	double general_helper[{{number_of_general_helpers}}];
	general(Y, general_helper);
	{% endif %}
	
	{% if number_of_jac_helpers>0: %}
	double jac_helper[{{number_of_jac_helpers}}];
	# include "jac_helpers.c"
	{% endif %}
	
	# include "jac.c"
	
	return PyArray_Return(dfdY);
}
{% endif %}

# pragma GCC diagnostic pop

static PyMethodDef {{module_name}}_methods[] = {
	{"f", py_f, METH_VARARGS, NULL},
	{% if has_Jacobian: %}
	{"jac", py_jac, METH_VARARGS, NULL},
	{% endif %}
	{NULL, NULL, 0, NULL}
};


{% if Python_version==3: %}

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

{% elif Python_version==2: %}

PyMODINIT_FUNC init{{module_name}}(void)
{
	Py_InitModule("{{module_name}}", {{module_name}}_methods);
	import_array();
}

{% endif %}