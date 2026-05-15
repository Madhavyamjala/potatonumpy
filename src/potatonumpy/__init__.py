"""PotatoNumPy — A pure Python linear algebra and tensor library.

Educational library demonstrating how array operations work internally,
using only Python lists, loops, and recursion. No external dependencies.
"""

from potatonumpy.core import PotatoNumPy, array
from potatonumpy.exceptions import (
    PotatoNumPyError,
    ShapeMismatchError,
    InvalidOperationError,
    InvalidTensorError,
    SingularMatrixError,
)
from potatonumpy.linalg import (
    dot,
    cross,
    matmul,
    transpose,
    magnitude,
    normalize,
    determinant,
    inverse,
    trace,
    identity,
    zeros,
    diagonal,
)
from potatonumpy.tensor import (
    tensor_sum,
    tensor_min,
    tensor_max,
    tensor_mean,
)
from potatonumpy.benchmarks import run_all_benchmarks

__version__ = "0.1.0"
__author__ = "Madhav"

__all__ = [
    "PotatoNumPy",
    "array",
    "dot",
    "cross",
    "matmul",
    "transpose",
    "magnitude",
    "normalize",
    "determinant",
    "inverse",
    "trace",
    "identity",
    "zeros",
    "diagonal",
    "tensor_sum",
    "tensor_min",
    "tensor_max",
    "tensor_mean",
    "run_all_benchmarks",
    "PotatoNumPyError",
    "ShapeMismatchError",
    "InvalidOperationError",
    "InvalidTensorError",
    "SingularMatrixError",
]
