from setuptools import setup
from io import open

requirements = [
	'jitcxde_common>=1.1.2',
	'symengine>=0.3.1.dev0',
	'scipy',
	'numpy'
]

setup(
	name = 'jitcode',
	description = 'Just-in-Time Compilation for Ordinary Differential Equations',
	long_description = open('README.rst', encoding='utf8').read(),
	author = 'Gerrit Ansmann',
	author_email = 'gansmann@uni-bonn.de',
	url = 'http://github.com/neurophysik/jitcode',
	python_requires=">=3.3",
	packages = ['jitcode'],
	package_data = {'jitcode': ['jitced_template.c']},
	include_package_data = True,
	install_requires = requirements,
	setup_requires = ['setuptools_scm'],
	use_scm_version = {'write_to': 'jitcode/version.py'},
	classifiers = [
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: BSD License',
		'Operating System :: POSIX',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
		'Programming Language :: Python',
		'Topic :: Scientific/Engineering :: Mathematics',
		],
)

