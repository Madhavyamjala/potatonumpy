"""Benchmarking utilities comparing PotatoNumPy vs pure loops vs NumPy."""

import time
from typing import List, Tuple, Optional


def _time_it(func, *args, repeats: int = 1) -> float:
    """Time a function call and return elapsed seconds."""
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    end = time.perf_counter()
    return end - start


def benchmark_vector_addition(size: int = 10000) -> dict:
    """Benchmark vector addition: pure loops vs PotatoNumPy vs NumPy."""
    from potatonumpy.core import PotatoNumPy

    list_a = [float(i) for i in range(size)]
    list_b = [float(i) for i in range(size)]

    def pure_loop_add():
        result = [0.0] * size
        for i in range(size):
            result[i] = list_a[i] + list_b[i]
        return result

    pp_a = PotatoNumPy(list_a)
    pp_b = PotatoNumPy(list_b)

    def potatonumpy_add():
        return pp_a + pp_b

    time_loop = _time_it(pure_loop_add)
    time_pp = _time_it(potatonumpy_add)

    results = {
        "operation": f"Vector Addition (size={size})",
        "pure_loop": time_loop,
        "potatonumpy": time_pp,
    }

    try:
        import numpy as np
        np_a = np.array(list_a)
        np_b = np.array(list_b)

        def numpy_add():
            return np_a + np_b

        time_np = _time_it(numpy_add)
        results["numpy"] = time_np
        results["pp_vs_numpy"] = time_pp / time_np if time_np > 0 else float("inf")
    except ImportError:
        results["numpy"] = None
        results["pp_vs_numpy"] = None

    return results


def benchmark_matrix_multiply(size: int = 50) -> dict:
    """Benchmark matrix multiplication: pure loops vs PotatoNumPy vs NumPy."""
    from potatonumpy.core import PotatoNumPy

    mat_a = [[float(i * size + j) for j in range(size)] for i in range(size)]
    mat_b = [[float(i * size + j) for j in range(size)] for i in range(size)]

    def pure_loop_matmul():
        result = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                total = 0.0
                for k in range(size):
                    total = total + mat_a[i][k] * mat_b[k][j]
                result[i][j] = total
        return result

    pp_a = PotatoNumPy(mat_a)
    pp_b = PotatoNumPy(mat_b)

    def potatonumpy_matmul():
        from potatonumpy.linalg import matmul
        return matmul(pp_a, pp_b)

    time_loop = _time_it(pure_loop_matmul)
    time_pp = _time_it(potatonumpy_matmul)

    results = {
        "operation": f"Matrix Multiplication ({size}x{size})",
        "pure_loop": time_loop,
        "potatonumpy": time_pp,
    }

    try:
        import numpy as np
        np_a = np.array(mat_a)
        np_b = np.array(mat_b)

        def numpy_matmul():
            return np_a @ np_b

        time_np = _time_it(numpy_matmul)
        results["numpy"] = time_np
        results["pp_vs_numpy"] = time_pp / time_np if time_np > 0 else float("inf")
    except ImportError:
        results["numpy"] = None
        results["pp_vs_numpy"] = None

    return results


def benchmark_scalar_operations(size: int = 10000) -> dict:
    """Benchmark scalar multiplication: pure loops vs PotatoNumPy vs NumPy."""
    from potatonumpy.core import PotatoNumPy

    data = [float(i) for i in range(size)]
    scalar = 3.14

    def pure_loop_scalar():
        result = [0.0] * size
        for i in range(size):
            result[i] = data[i] * scalar
        return result

    pp_arr = PotatoNumPy(data)

    def potatonumpy_scalar():
        return pp_arr * scalar

    time_loop = _time_it(pure_loop_scalar)
    time_pp = _time_it(potatonumpy_scalar)

    results = {
        "operation": f"Scalar Multiplication (size={size})",
        "pure_loop": time_loop,
        "potatonumpy": time_pp,
    }

    try:
        import numpy as np
        np_arr = np.array(data)

        def numpy_scalar():
            return np_arr * scalar

        time_np = _time_it(numpy_scalar)
        results["numpy"] = time_np
        results["pp_vs_numpy"] = time_pp / time_np if time_np > 0 else float("inf")
    except ImportError:
        results["numpy"] = None
        results["pp_vs_numpy"] = None

    return results


def print_benchmark(results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  {results['operation']}")
    print(f"{'=' * 60}")
    print(f"  Pure Python loops : {results['pure_loop']:.6f}s")
    print(f"  PotatoNumPy       : {results['potatonumpy']:.6f}s")

    if results.get("numpy") is not None:
        print(f"  NumPy             : {results['numpy']:.6f}s")
        print(f"  PP/NumPy slowdown : {results['pp_vs_numpy']:.1f}x")
    else:
        print(f"  NumPy             : (not installed)")

    print(f"{'=' * 60}")


def run_all_benchmarks() -> None:
    """Run all benchmarks and print results."""
    print("\n" + "=" * 60)
    print("  PotatoNumPy Performance Benchmarks")
    print("=" * 60)

    benchmarks = [
        benchmark_vector_addition,
        benchmark_matrix_multiply,
        benchmark_scalar_operations,
    ]

    for bench_func in benchmarks:
        results = bench_func()
        print_benchmark(results)

    print("\nKey Takeaway: NumPy uses C extensions and vectorized operations")
    print("that bypass Python's interpreter overhead, making it orders of")
    print("magnitude faster than pure Python implementations.\n")
