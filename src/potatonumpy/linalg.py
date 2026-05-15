"""Linear algebra operations for PotatoNumPy arrays.

All algorithms are implemented with explicit Python loops and recursion
to demonstrate how these operations work internally.
"""

from __future__ import annotations
import math
from typing import List, Any, TYPE_CHECKING

from potatonumpy.exceptions import (
    ShapeMismatchError,
    InvalidOperationError,
    SingularMatrixError,
)
from potatonumpy.utils import is_numeric

if TYPE_CHECKING:
    from potatonumpy.core import PotatoNumPy


def _ensure_vector(arr: PotatoNumPy, name: str = "array") -> List:
    if arr.ndim != 1:
        raise InvalidOperationError(f"{name} must be a 1D vector, got {arr.ndim}D")
    return arr.data


def _ensure_matrix(arr: PotatoNumPy, name: str = "array") -> List[List]:
    if arr.ndim != 2:
        raise InvalidOperationError(f"{name} must be a 2D matrix, got {arr.ndim}D")
    return arr.data


def _ensure_square(arr: PotatoNumPy, name: str = "array") -> List[List]:
    data = _ensure_matrix(arr, name)
    rows, cols = arr.shape
    if rows != cols:
        raise InvalidOperationError(
            f"{name} must be square, got shape {arr.shape}"
        )
    return data


def dot(a: PotatoNumPy, b: PotatoNumPy) -> Any:
    """Compute the dot product of two vectors.

    Iterates element-by-element, multiplying and accumulating the sum
    using a plain Python loop.
    """
    vec_a = _ensure_vector(a, "first argument")
    vec_b = _ensure_vector(b, "second argument")

    if len(vec_a) != len(vec_b):
        raise ShapeMismatchError(a.shape, b.shape, "dot product")

    result = 0
    for i in range(len(vec_a)):
        result = result + vec_a[i] * vec_b[i]
    return result


def cross(a: PotatoNumPy, b: PotatoNumPy) -> PotatoNumPy:
    """Compute the cross product of two 3D vectors.

    Uses the explicit determinant formula for the 3x3 case.
    """
    from potatonumpy.core import PotatoNumPy as PP

    vec_a = _ensure_vector(a, "first argument")
    vec_b = _ensure_vector(b, "second argument")

    if len(vec_a) != 3 or len(vec_b) != 3:
        raise InvalidOperationError("Cross product requires 3D vectors")

    result = [
        vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1],
        vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2],
        vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0],
    ]
    return PP(result)


def matmul(a: PotatoNumPy, b: PotatoNumPy) -> PotatoNumPy:
    """Multiply two matrices using three nested Python loops.

    Implements the textbook O(n^3) matrix multiplication algorithm:
    C[i][j] = sum(A[i][k] * B[k][j] for k in range(K))
    """
    from potatonumpy.core import PotatoNumPy as PP

    mat_a = _ensure_matrix(a, "first argument")
    mat_b = _ensure_matrix(b, "second argument")

    rows_a, cols_a = a.shape
    rows_b, cols_b = b.shape

    if cols_a != rows_b:
        raise ShapeMismatchError(
            a.shape, b.shape, "matrix multiplication"
        )

    result = []
    for i in range(rows_a):
        row = []
        for j in range(cols_b):
            total = 0
            for k in range(cols_a):
                total = total + mat_a[i][k] * mat_b[k][j]
            row.append(total)
        result.append(row)

    return PP(result)


def transpose(a: PotatoNumPy) -> PotatoNumPy:
    """Transpose a 2D matrix by swapping rows and columns.

    Builds the result matrix element-by-element with explicit loops.
    """
    from potatonumpy.core import PotatoNumPy as PP

    mat = _ensure_matrix(a)
    rows, cols = a.shape

    result = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(mat[i][j])
        result.append(new_row)

    return PP(result)


def magnitude(a: PotatoNumPy) -> float:
    """Compute the Euclidean magnitude (L2 norm) of a vector."""
    vec = _ensure_vector(a)
    total = 0
    for i in range(len(vec)):
        total = total + vec[i] * vec[i]
    return math.sqrt(total)


def normalize(a: PotatoNumPy) -> PotatoNumPy:
    """Return a unit vector in the same direction as the input."""
    from potatonumpy.core import PotatoNumPy as PP

    vec = _ensure_vector(a)
    mag = magnitude(a)

    if mag == 0:
        raise InvalidOperationError("Cannot normalize a zero vector")

    result = [None] * len(vec)
    for i in range(len(vec)):
        result[i] = vec[i] / mag

    return PP(result)


def determinant(a: PotatoNumPy) -> float:
    """Compute the determinant of a square matrix using cofactor expansion.

    Recursively expands along the first row, computing minors at each step.
    This is the classic O(n!) algorithm — intentionally slow to demonstrate
    the cost of naive recursion.
    """
    mat = _ensure_square(a, "matrix")
    return _det_recursive(mat)


def _det_recursive(mat: List[List]) -> float:
    """Recursive cofactor expansion along the first row."""
    n = len(mat)

    if n == 1:
        return mat[0][0]

    if n == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    det = 0
    for col in range(n):
        minor = _get_minor(mat, 0, col)
        sign = 1 if col % 2 == 0 else -1
        det = det + sign * mat[0][col] * _det_recursive(minor)

    return det


def _get_minor(mat: List[List], row: int, col: int) -> List[List]:
    """Extract the minor matrix by removing the specified row and column."""
    n = len(mat)
    result = []
    for i in range(n):
        if i == row:
            continue
        new_row = []
        for j in range(n):
            if j == col:
                continue
            new_row.append(mat[i][j])
        result.append(new_row)
    return result


def inverse(a: PotatoNumPy) -> PotatoNumPy:
    """Compute the inverse of a square matrix using the adjugate method.

    Calculates the matrix of cofactors, transposes it to get the adjugate,
    and divides by the determinant. Raises SingularMatrixError if det = 0.
    """
    from potatonumpy.core import PotatoNumPy as PP

    mat = _ensure_square(a, "matrix")
    n = len(mat)
    det = _det_recursive(mat)

    if abs(det) < 1e-12:
        raise SingularMatrixError()

    if n == 1:
        return PP([[1.0 / mat[0][0]]])

    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = _get_minor(mat, i, j)
            sign = 1 if (i + j) % 2 == 0 else -1
            cofactor = sign * _det_recursive(minor)
            row.append(cofactor)
        cofactor_matrix.append(row)

    adjugate = []
    for j in range(n):
        row = []
        for i in range(n):
            row.append(cofactor_matrix[i][j] / det)
        adjugate.append(row)

    return PP(adjugate)


def trace(a: PotatoNumPy) -> float:
    """Compute the trace (sum of diagonal elements) of a square matrix."""
    mat = _ensure_square(a, "matrix")
    n = len(mat)
    total = 0
    for i in range(n):
        total = total + mat[i][i]
    return total


def identity(n: int) -> PotatoNumPy:
    """Create an n x n identity matrix."""
    from potatonumpy.core import PotatoNumPy as PP

    if not isinstance(n, int) or n <= 0:
        raise InvalidOperationError("Identity matrix size must be a positive integer")

    result = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        result.append(row)

    return PP(result)


def zeros(shape: tuple) -> PotatoNumPy:
    """Create a matrix of zeros with the given shape."""
    from potatonumpy.core import PotatoNumPy as PP

    if not isinstance(shape, tuple) or len(shape) < 1:
        raise InvalidOperationError("Shape must be a non-empty tuple")

    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise InvalidOperationError("All dimensions must be positive integers")

    return PP(_build_zeros(shape))


def _build_zeros(shape: tuple) -> Any:
    """Recursively build a nested list of zeros."""
    if len(shape) == 1:
        result = [0] * shape[0]
        return result

    result = []
    remaining = shape[1:]
    for i in range(shape[0]):
        result.append(_build_zeros(remaining))
    return result


def diagonal(a: PotatoNumPy) -> PotatoNumPy:
    """Extract the diagonal elements of a square matrix as a vector."""
    from potatonumpy.core import PotatoNumPy as PP

    mat = _ensure_square(a, "matrix")
    n = len(mat)

    result = [None] * n
    for i in range(n):
        result[i] = mat[i][i]

    return PP(result)
