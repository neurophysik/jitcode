import sys
import os
from unittest.mock import MagicMock
from setuptools_scm import get_version

class Mock(MagicMock):
	@classmethod
	def __getattr__(cls, name):
		return MagicMock()

MOCK_MODULES = [
	'numpy', 'numpy.testing', 'numpy.random',
	'scipy', 'scipy.integrate', 'scipy.integrate._ode', 'scipy.stats',
	'sympy',
	'jitcode', 'jitcode._helpers']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

sys.modules['scipy.integrate'] = Mock(ode=object)

sys.path.insert(0,os.path.abspath("../examples"))
sys.path.insert(0,os.path.abspath("../jitcode"))

needs_sphinx = '1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.graphviz'
]

source_suffix = '.rst'

master_doc = 'index'

project = u'JiTCODE'
copyright = u'2016, Gerrit Ansmann'

release = version = get_version(root='..', relative_to=__file__)

default_role = "any"

add_function_parentheses = True

add_module_names = False

html_theme = 'pyramid'
pygments_style = 'colorful'
#html_theme_options = {}
htmlhelp_basename = 'JiTCODEdoc'

numpydoc_show_class_members = False
autodoc_member_order = 'bysource'

graphviz_output_format = "svg"