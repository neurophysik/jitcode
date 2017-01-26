from __future__ import print_function, division, with_statement
from jinja2 import Environment, FileSystemLoader
from sympy.printing.ccode import ccode
from sys import version_info, stderr
import numpy as np
from os import path
from warnings import warn
from itertools import chain
from sympy.core.cache import clear_cache
from sympy import __version__ as sympy_version


# String manipulation
# -------------------

def remove_suffix(string, suffix):
	partition = string.rpartition(suffix)
	if partition[1] and not partition[2]:
		return partition[0]
	else:
		return string

def ensure_suffix(string, suffix):
	if not string.endswith(suffix):
		return string + suffix
	else:
		return string

def rsplit_int(s):
	if s and s[-1].isdigit():
		x,y = rsplit_int(s[:-1])
		return x, y+s[-1]
	else:
		return s, ""

def count_up(name):
	s, i = rsplit_int(name)
	return s + ( "%%.%ii" % len(i) % (int(i)+1)  if i else "_1" )

# Module handling
# ---------------

if version_info < (3,):
	
	from imp import find_module, load_module, load_dynamic
	
	def modulename_from_path(full_path):
		filename = path.basename(full_path)
		return remove_suffix(filename, ".so")
	
	def get_module_path(modulename, folder=""):
		return find_module(modulename, [folder])[1]
	
	def find_and_load_module(modulename, folder=""):
		specs = find_module(modulename, [folder])
		return load_module(modulename, *specs)
	
	def module_from_path(full_path):
		return load_dynamic( modulename_from_path(full_path), full_path )
	
elif version_info >= (3,3):
	
	from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES, FileFinder
	from importlib.util import spec_from_file_location
	
	loader_details = (ExtensionFileLoader, EXTENSION_SUFFIXES)
	
	def modulename_from_path(full_path):
		filename = path.basename(full_path)
		for suffix in sorted(EXTENSION_SUFFIXES, key=len, reverse=True):
			filename = remove_suffix(filename, suffix)
		return filename
	
	def get_module_path(modulename, folder=""):
		finder = FileFinder(folder, loader_details)
		return finder.find_spec(modulename).origin
	
	if version_info < (3,5):
		def find_and_load_module(modulename, folder=""):
			finder = FileFinder(folder, loader_details)
			spec = finder.find_spec(modulename)
			return spec.loader.load_module()
		
		def module_from_path(full_path):
			modulename = modulename_from_path(full_path)
			spec = spec_from_file_location(modulename, full_path)
			return spec.loader.load_module()
		
	else:
		from importlib.util import module_from_spec
		
		def find_and_load_module(modulename, folder=""):
			finder = FileFinder(folder, loader_details)
			spec = finder.find_spec(modulename)
			module = module_from_spec(spec)
			spec.loader.exec_module(module)
			return module
		
		def module_from_path(full_path):
			modulename = modulename_from_path(full_path)
			spec = spec_from_file_location(modulename, full_path)
			module = module_from_spec(spec)
			spec.loader.exec_module(module)
			return module
	
else:
	
	raise NotImplementedError("Module loading for Python versions between 3 and 3.3 was not implemented. Please upgrade to a newer Python version.")

# Code and templates
# ------------------

def check_code(code):
	if code.startswith("// Not"):
		stderr.write(code)
		raise Exception("The above expression could not be converted to C Code.")
	return code

def write_in_chunks(lines, mainfile, deffile, name, chunk_size, arguments):
	funcname = "definitions_" + name
	
	first_chunk = []
	try:
		for i in range(chunk_size+1):
			first_chunk.append(next(lines))
	except StopIteration:
		for line in first_chunk:
			mainfile.write(line)
	else:
		lines = chain(first_chunk, lines)
		
		if sympy_version >= "1":
			clear_sympy_cache = clear_cache
		else:
			def clear_sympy_cache(): pass
			warn("Not clearing SymPy cache between chunks because this is buggy in this SymPy version. If excessive memory is used, this is why and you have to upgrade SymPy.")
		
		while True:
			mainfile.write(funcname + "(")
			deffile.write("void " + funcname + "(")
			if arguments:
				mainfile.write(", ".join(argument[0] for argument in arguments))
				deffile.write(", ".join(argument[1]+" "+argument[0] for argument in arguments))
			else:
				deffile.write("void")
			mainfile.write(");\n")
			deffile.write("){\n")
			
			try:
				for i in range(chunk_size):
					deffile.write(next(lines))
			except StopIteration:
				break
			finally:
				deffile.write("}\n")
			
			funcname = count_up(funcname)
			clear_sympy_cache()

def render_and_write_code(
	expressions,
	tmpfile,
	name,
	functions = [],
	chunk_size = 100,
	arguments = []
	):
	
	user_functions = {function:function for function in functions}
	
	def codelines():
		for expression in expressions:
			codeline = ccode(expression, user_functions=user_functions)
			yield check_code(codeline) + ";\n"
	
	with \
		open( tmpfile(name+".c"            ), "w" ) as mainfile, \
		open( tmpfile(name+"_definitions.c"), "w" ) as deffile:
		
		if chunk_size < 1:
			for line in codelines():
				mainfile.write(line)
		else:
			write_in_chunks(codelines(), mainfile, deffile, name, chunk_size, arguments)

def render_template(filename, target, **kwargs):
	folder = path.dirname(__file__)
	env = Environment(loader=FileSystemLoader(folder))
	template = env.get_template(filename)
	with open(target, "w") as codefile:
		codefile.write(template.render(kwargs))


# Numerical tools
# ---------------

# Return the ratio of zero entries in a SymPy matrix
def non_zero_ratio(A):
	return sum(x!=0 for x in A) / len(A)

def random_direction(n):
	vector = np.random.normal(0,1,n)
	return vector/np.linalg.norm(vector)

def orthonormalise(vectors):
	"""
	Orthonormalise vectors (with Gram-Schmidt) and return their norms after orthogonalisation (but before normalisation).
	"""
	norms = []
	for i,vector in enumerate(vectors):
		for j in range(i):
			vector -= np.dot( vector, vectors[j] ) * vectors[j]
		norm = np.linalg.norm(vector)
		vector /= norm
		norms.append(norm)
	
	return np.array(norms)
