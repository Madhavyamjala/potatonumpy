"""Core PotatoNumPy array class with elementwise operations and broadcasting."""

from __future__ import annotations
from typing import Any, List, Tuple, Union

from potatonumpy.exceptions import (
    ShapeMismatchError,
    InvalidOperationError,
    InvalidTensorError,
)
from potatonumpy.utils import (
    Numeric,
    is_numeric,
    validate_tensor,
    compute_shape,
    deep_copy,
    recursive_apply,
    recursive_elementwise,
    flatten,
    reshape_flat,
)


class PotatoNumPy:
    """N-dimensional array class mimicking NumPy's ndarray.

    Stores data as nested Python lists and provides elementwise arithmetic,
    shape inspection, and broadcasting with scalar values.
    """

    def __init__(self, data: Any) -> None:
        if is_numeric(data):
            self._data = data
            self._shape: Tuple[int, ...] = ()
            self._ndim: int = 0
        elif isinstance(data, list):
            validate_tensor(data)
            self._data = deep_copy(data)
            self._shape = compute_shape(data)
            self._ndim = len(self._shape)
        else:
            raise InvalidTensorError(
                f"Cannot create array from type {type(data).__name__}"
            )

        total = 1
        for dim in self._shape:
            total = total * dim
        self._size: int = total

    @property
    def data(self) -> Any:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self) -> int:
        return self._size

    def tolist(self) -> Any:
        """Return the array data as a nested Python list."""
        return deep_copy(self._data)

    def flatten(self) -> PotatoNumPy:
        """Return a 1D copy of the array."""
        return PotatoNumPy(flatten(self._data))

    def reshape(self, new_shape: Tuple[int, ...]) -> PotatoNumPy:
        """Return a reshaped copy of the array."""
        flat = flatten(self._data)
        return PotatoNumPy(reshape_flat(flat, new_shape))

    def _broadcast_op(
        self, other: Any, op_name: str, op_func
    ) -> PotatoNumPy:
        """Perform an elementwise operation with scalar broadcasting."""
        if isinstance(other, PotatoNumPy):
            if self._shape != other._shape:
                raise ShapeMismatchError(self._shape, other._shape, op_name)
            result = recursive_elementwise(self._data, other._data, op_func)
            return PotatoNumPy(result)

        if is_numeric(other):
            result = recursive_apply(self._data, lambda x: op_func(x, other))
            return PotatoNumPy(result)

        raise InvalidOperationError(
            f"Unsupported operand type for {op_name}: {type(other).__name__}"
        )

    def _r_broadcast_op(
        self, other: Any, op_name: str, op_func
    ) -> PotatoNumPy:
        """Perform a reflected elementwise operation with scalar broadcasting."""
        if is_numeric(other):
            result = recursive_apply(self._data, lambda x: op_func(other, x))
            return PotatoNumPy(result)

        raise InvalidOperationError(
            f"Unsupported operand type for {op_name}: {type(other).__name__}"
        )

    def __add__(self, other: Any) -> PotatoNumPy:
        return self._broadcast_op(other, "addition", lambda a, b: a + b)

    def __radd__(self, other: Any) -> PotatoNumPy:
        return self._r_broadcast_op(other, "addition", lambda a, b: a + b)

    def __sub__(self, other: Any) -> PotatoNumPy:
        return self._broadcast_op(other, "subtraction", lambda a, b: a - b)

    def __rsub__(self, other: Any) -> PotatoNumPy:
        return self._r_broadcast_op(other, "subtraction", lambda a, b: a - b)

    def __mul__(self, other: Any) -> PotatoNumPy:
        return self._broadcast_op(other, "multiplication", lambda a, b: a * b)

    def __rmul__(self, other: Any) -> PotatoNumPy:
        return self._r_broadcast_op(other, "multiplication", lambda a, b: a * b)

    def __truediv__(self, other: Any) -> PotatoNumPy:
        def safe_div(a: Numeric, b: Numeric) -> Numeric:
            if b == 0:
                raise InvalidOperationError("Division by zero")
            return a / b

        return self._broadcast_op(other, "division", safe_div)

    def __rtruediv__(self, other: Any) -> PotatoNumPy:
        def safe_div(a: Numeric, b: Numeric) -> Numeric:
            if b == 0:
                raise InvalidOperationError("Division by zero")
            return a / b

        return self._r_broadcast_op(other, "division", safe_div)

    def __pow__(self, other: Any) -> PotatoNumPy:
        return self._broadcast_op(other, "power", lambda a, b: a ** b)

    def __rpow__(self, other: Any) -> PotatoNumPy:
        return self._r_broadcast_op(other, "power", lambda a, b: a ** b)

    def __mod__(self, other: Any) -> PotatoNumPy:
        def safe_mod(a: Numeric, b: Numeric) -> Numeric:
            if b == 0:
                raise InvalidOperationError("Modulus by zero")
            return a % b

        return self._broadcast_op(other, "modulus", safe_mod)

    def __neg__(self) -> PotatoNumPy:
        return PotatoNumPy(recursive_apply(self._data, lambda x: -x))

    def __pos__(self) -> PotatoNumPy:
        return PotatoNumPy(deep_copy(self._data))

    def __abs__(self) -> PotatoNumPy:
        return PotatoNumPy(recursive_apply(self._data, lambda x: abs(x)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PotatoNumPy):
            return NotImplemented
        if self._shape != other._shape:
            return False
        from potatonumpy.utils import recursive_equals
        return recursive_equals(self._data, other._data)

    def __getitem__(self, index: Any) -> Any:
        if self._ndim == 0:
            raise InvalidOperationError("Cannot index a scalar array")

        result = self._data[index]
        if is_numeric(result):
            return result
        return PotatoNumPy(result)

    def __len__(self) -> int:
        if self._ndim == 0:
            raise InvalidOperationError("Scalar array has no length")
        return self._shape[0]

    def _format_data(self, data: Any, depth: int = 0) -> str:
        if is_numeric(data):
            if isinstance(data, float):
                return f"{data:.4f}"
            return str(data)

        if not isinstance(data, list):
            return str(data)

        indent = " " * depth
        inner_indent = " " * (depth + 1)

        if len(data) == 0:
            return "[]"

        if is_numeric(data[0]):
            elements = []
            for i in range(len(data)):
                elements.append(self._format_data(data[i], depth + 1))
            return "[" + ", ".join(elements) + "]"

        lines = []
        lines.append("[")
        for i in range(len(data)):
            formatted = self._format_data(data[i], depth + 1)
            if i < len(data) - 1:
                lines.append(inner_indent + formatted + ",")
            else:
                lines.append(inner_indent + formatted)
        lines.append(indent + "]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"PotatoNumPy({self._format_data(self._data)})"

    def __repr__(self) -> str:
        return f"PotatoNumPy(data={self._data}, shape={self._shape})"


def array(data: Any) -> PotatoNumPy:
    """Create a PotatoNumPy array from a scalar, list, or nested list."""
    return PotatoNumPy(data)
