from ._jitcode import (
		t, y,
		jitcode, jitcode_lyap, jitcode_restricted_lyap, jitcode_transversal_lyap,
		UnsuccessfulIntegration,
		test,
		)

try:
	from .version import version as __version__
except ImportError:
	from warnings import warn
	warn('Failed to find (autogenerated) version.py. Do not worry about this unless you really need to know the version.')
