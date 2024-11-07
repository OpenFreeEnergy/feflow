"""Module with custom exceptions that can be useful for including in protocols"""

# TODO: We should implement the base class in gufe.
#  Issue https://github.com/OpenFreeEnergy/gufe/issues/385
class ProtocolError(Exception):
    """Base exception for OpenFE protocols"""


class MethodLimitationtError(ProtocolError):
    """Custom exception to be raised when a fundamental limitation in the methodology is
    that does not support."""


class NotSupportedError(ProtocolError):
    """Custom exception to be raised when a specific scenario or case is not supported by the
    protocol."""
