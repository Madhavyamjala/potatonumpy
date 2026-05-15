"""Custom exceptions for the PotatoNumPy library."""


class PotatoNumPyError(Exception):
    """Base exception for all PotatoNumPy errors."""

    pass


class ShapeMismatchError(PotatoNumPyError):
    """Raised when array shapes are incompatible for an operation."""

    def __init__(self, shape_a: tuple, shape_b: tuple, operation: str = "operation"):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.operation = operation
        super().__init__(f"Shape mismatch for {operation}: {shape_a} vs {shape_b}")


class InvalidOperationError(PotatoNumPyError):
    """Raised when an operation is not valid for the given array."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidTensorError(PotatoNumPyError):
    """Raised when a tensor has an invalid or inconsistent structure."""

    def __init__(
        self, message: str = "Invalid tensor structure: inconsistent dimensions"
    ):
        super().__init__(message)


class SingularMatrixError(PotatoNumPyError):
    """Raised when a matrix is singular and cannot be inverted."""

    def __init__(self, message: str = "Matrix is singular and cannot be inverted"):
        super().__init__(message)
