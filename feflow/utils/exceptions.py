"""Module with custom exceptions that can be useful for including in protocols"""


# TODO: We should implement the base class in gufe.
#  Issue https://github.com/OpenFreeEnergy/gufe/issues/385
class ProtocolError(Exception):
    """Base exception for OpenFE protocols"""


class MethodLimitationtError(ProtocolError):
    """Custom exception raised when a fundamental limitation in the methodology prevents support
    for the requested operation."""


class ProtocolSupportError(ProtocolError):
    """Custom exception raised when the tooling does not support a specific scenario or use case."""
