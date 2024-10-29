"""Module with custom exceptions that can be useful for including in protocols"""


class ProtocolError(Exception):
    """Base exception for OpenFE protocols"""


class MethodConstraintError(ProtocolError):
    """Custom exception to be raised when a fundamental limitation in the methodology is
    that does not support."""


class NotSupportedError(ProtocolError):
    """Custom exception to be raised when a specific scenario or case is not supported by the
    protocol."""
