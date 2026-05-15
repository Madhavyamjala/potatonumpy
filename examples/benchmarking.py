"""Examples: Performance benchmarking with PotatoNumPy."""

import potatonumpy as pp


def main():
    print("=" * 60)
    print("  PotatoNumPy vs NumPy Performance Comparison")
    print("=" * 60)
    print()
    print("This benchmark demonstrates why NumPy is dramatically")
    print("faster than pure Python implementations.")
    print()
    print("NumPy advantages:")
    print("  - C/Fortran compiled inner loops")
    print("  - SIMD vectorization")
    print("  - Cache-friendly memory layout")
    print("  - BLAS/LAPACK for linear algebra")
    print()
    print("PotatoNumPy intentionally uses:")
    print("  - Pure Python loops (interpreter overhead)")
    print("  - Recursive algorithms (call stack overhead)")
    print("  - Python lists (no contiguous memory)")
    print("  - Dynamic type checking at every step")
    print()

    pp.run_all_benchmarks()


if __name__ == "__main__":
    main()
