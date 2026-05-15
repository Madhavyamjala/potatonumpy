"""Examples: Vector operations with PotatoNumPy."""

import potatonumpy as pp


def main():
    print("=" * 50)
    print("  PotatoNumPy Vector Examples")
    print("=" * 50)

    a = pp.array([1, 2, 3])
    b = pp.array([4, 5, 6])

    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"Shape: {a.shape}")

    print(f"\na + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"a + 10 = {a + 10}")
    print(f"3 * a = {3 * a}")

    print(f"\nDot product: {pp.dot(a, b)}")

    i = pp.array([1, 0, 0])
    j = pp.array([0, 1, 0])
    print(f"\ni x j = {pp.cross(i, j)}")

    print(f"\nMagnitude of a: {pp.magnitude(a):.4f}")
    print(f"Normalized a: {pp.normalize(a)}")

    c = pp.array([1 + 2j, 3 + 4j])
    print(f"\nComplex vector: {c}")
    print(f"Complex + 1: {c + 1}")


if __name__ == "__main__":
    main()
