"""Examples: Matrix operations with PotatoNumPy."""

import potatonumpy as pp


def main():
    print("=" * 50)
    print("  PotatoNumPy Matrix Examples")
    print("=" * 50)

    a = pp.array([[1, 2], [3, 4]])
    b = pp.array([[5, 6], [7, 8]])

    print(f"\nA =\n{a}")
    print(f"\nB =\n{b}")

    print(f"\nA + B =\n{a + b}")
    print(f"\nA * B (elementwise) =\n{a * b}")
    print(f"\nA @ B (matmul) =\n{pp.matmul(a, b)}")

    print(f"\nTranspose of A =\n{pp.transpose(a)}")
    print(f"\nDeterminant of A = {pp.determinant(a)}")
    print(f"Trace of A = {pp.trace(a)}")
    print(f"Diagonal of A = {pp.diagonal(a)}")

    print(f"\nInverse of A =\n{pp.inverse(a)}")

    product = pp.matmul(a, pp.inverse(a))
    print(f"\nA @ inv(A) =\n{product}")

    print(f"\nIdentity(3) =\n{pp.identity(3)}")
    print(f"\nZeros(2,3) =\n{pp.zeros((2, 3))}")

    c = pp.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    print(f"\nC =\n{c}")
    print(f"Determinant of C = {pp.determinant(c)}")
    print(f"\nInverse of C =\n{pp.inverse(c)}")


if __name__ == "__main__":
    main()
