# PotatoNumPy 🥔

A pure Python linear algebra and tensor library built entirely from scratch. No NumPy, no C extensions, no shortcuts — just potatoes all the way down.

> It took me 3 weeks alone and 2 hours of Claude documentation refinement to make this.

## What is this?

PotatoNumPy is a learning project that answers three questions I kept asking myself:

1. **How does linear algebra actually work under the hood?** Every matrix multiplication here is three nested loops. Every determinant is recursive cofactor expansion. Nothing hides behind an opaque C function — you can read every line and follow along.

2. **Why is NumPy so ridiculously fast?** After watching PotatoNumPy chug through a 50x50 matrix multiply, you'll *feel* what NumPy's C/Fortran backends, SIMD vectorization, and contiguous memory layouts buy you. It's not subtle.

3. **What does Python's interpreter overhead actually cost?** Dynamic type checking, function call overhead, pointer-chasing through Python lists — it all adds up. This library makes that cost painfully tangible.

## Installation

```bash
pip install potatonumpy
```

Or clone and install locally:

```bash
git clone https://github.com/Madhavyamjala/potatonumpy.git
cd potatonumpy
pip install -e .
```

No dependencies needed — that's the whole point.

## Quick Start

```python
import potatonumpy as pp

# Create arrays
a = pp.array([1, 2, 3])
b = pp.array([4, 5, 6])

# Elementwise math, just like you'd expect
print(a + b)        # [5, 7, 9]
print(a * b)        # [4, 10, 18]
print(a ** 2)       # [1, 4, 9]
print(a + 10)       # [11, 12, 13]

# Linear algebra
print(pp.dot(a, b))         # 32
print(pp.cross(a, b))       # [-3, 6, -3]
print(pp.magnitude(a))      # 3.7416...
print(pp.normalize(a))      # [0.2672, 0.5345, 0.8017]
```

## Matrix Operations

```python
import potatonumpy as pp

A = pp.array([[1, 2], [3, 4]])
B = pp.array([[5, 6], [7, 8]])

print(pp.matmul(A, B))      # [[19, 22], [43, 50]]
print(pp.transpose(A))      # [[1, 3], [2, 4]]
print(pp.determinant(A))    # -2.0
print(pp.inverse(A))        # [[-2, 1], [1.5, -0.5]]
print(pp.trace(A))          # 5
print(pp.diagonal(A))       # [1, 4]

# Special matrices
print(pp.identity(3))
print(pp.zeros((2, 3)))
```

## Tensors? Yeah, We Got Those

```python
import potatonumpy as pp

# 3D tensor — go wild
t = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t.shape)              # (2, 2, 2)

# Reductions
print(pp.tensor_sum(t))             # 36
print(pp.tensor_sum(t, axis=0))     # [[6, 8], [10, 12]]
print(pp.tensor_mean(t))            # 4.5
print(pp.tensor_min(t))             # 1
print(pp.tensor_max(t))             # 8
```

## Type Support

Works with `int`, `float`, and `complex`:

```python
c = pp.array([1+2j, 3+4j])
print(c + 1)    # [(2+2j), (4+4j)]
```

## Error Handling

PotatoNumPy doesn't silently do the wrong thing. It yells at you with specific exceptions:

```python
from potatonumpy import (
    ShapeMismatchError,  # tried to add (2,) and (3,)?
    InvalidTensorError,  # ragged array? nope
    SingularMatrixError,  # can't invert that matrix, sorry
    InvalidOperationError,  # division by zero, etc.
)

pp.array([1, 2]) + pp.array([1, 2, 3])  # ShapeMismatchError
pp.array([[1, 2], [3, 4, 5]])  # InvalidTensorError
pp.inverse(pp.array([[1, 2], [2, 4]]))  # SingularMatrixError
pp.array([1, 2]) / 0  # InvalidOperationError
```

## Benchmarks (Prepare to Cringe)

```bash
python examples/benchmarking.py
```

Here's what you'll see (roughly):

| Operation | Pure Loops | PotatoNumPy | NumPy | Slowdown vs NumPy |
|---|---|---|---|---|
| Vector Add (10k) | ~2ms | ~5ms | ~0.01ms | ~500x |
| Matrix Mul (50x50) | ~50ms | ~80ms | ~0.05ms | ~1600x |
| Scalar Mul (10k) | ~1ms | ~3ms | ~0.005ms | ~600x |

Yeah. NumPy is *that* much faster. Here's why:

| What | PotatoNumPy | NumPy |
|---|---|---|
| Inner loops | Python bytecode | Compiled C/Fortran |
| Memory layout | Scattered Python objects | Contiguous typed arrays |
| Vectorization | Nope | SIMD instructions |
| Type checking | Every single operation | Once at array creation |
| Function calls | Python stack frames | Inlined C calls |
| Math backend | Hand-rolled loops | BLAS/LAPACK |

## Running Tests

```bash
python -m unittest discover tests -v
```

100 tests covering all operations, edge cases, and error conditions.

## Project Structure

```
pythonPy/
├── potatonumpy/
│   ├── __init__.py       # Public API — import potatonumpy as pp
│   ├── core.py           # The PotatoNumPy array class
│   ├── linalg.py         # All the linear algebra goodness
│   ├── tensor.py         # Tensor reductions (sum, min, max, mean)
│   ├── exceptions.py     # Custom exceptions
│   ├── utils.py          # Recursive helpers
│   └── benchmarks.py     # Performance showdown
├── tests/
│   ├── test_core.py
│   ├── test_linalg.py
│   └── test_tensor.py
├── examples/
│   ├── vectors.py
│   ├── matrices.py
│   └── benchmarking.py
├── README.md
├── pyproject.toml
└── setup.py
```

## Design Philosophy

- **No magic.** Every operation is explicit loops you can step through in a debugger. Want to understand how matrix inverse works? Read the code.
- **Recursion over cleverness.** Determinants use cofactor expansion. Shape validation walks the full tree. This is intentional — clarity over performance.
- **The slowness is a feature.** Seriously. The massive performance gap between PotatoNumPy and NumPy *is* the lesson.
- **Zero dependencies.** Standard library only. If Python doesn't ship it, we don't use it.

## Limitations (by Design)

- **Slow.** Orders of magnitude slower than NumPy. That's the point.
- **Scalar broadcasting only.** No fancy NumPy-style shape broadcasting.
- **Memory hungry.** Python lists use ~8x more memory per element than NumPy arrays.
- **Determinant is O(n!).** Don't try matrices bigger than ~12x12 unless you have time to spare.
- **No eigenvalues, SVD, FFT, or sparse matrices.** This is an educational tool, not a production library.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with love, frustration, and an unreasonable number of nested for loops.* 🥔
