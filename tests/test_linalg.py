"""Unit tests for PotatoNumPy linear algebra operations."""

import unittest
import math

import potatonumpy as pp
from potatonumpy.exceptions import (
    ShapeMismatchError,
    InvalidOperationError,
    SingularMatrixError,
)


class TestDotProduct(unittest.TestCase):

    def test_basic_dot(self):
        a = pp.array([1, 2, 3])
        b = pp.array([4, 5, 6])
        self.assertEqual(pp.dot(a, b), 32)

    def test_dot_orthogonal(self):
        a = pp.array([1, 0, 0])
        b = pp.array([0, 1, 0])
        self.assertEqual(pp.dot(a, b), 0)

    def test_dot_shape_mismatch(self):
        a = pp.array([1, 2])
        b = pp.array([1, 2, 3])
        with self.assertRaises(ShapeMismatchError):
            pp.dot(a, b)

    def test_dot_non_vector(self):
        a = pp.array([[1, 2], [3, 4]])
        b = pp.array([1, 2])
        with self.assertRaises(InvalidOperationError):
            pp.dot(a, b)


class TestCrossProduct(unittest.TestCase):

    def test_basic_cross(self):
        a = pp.array([1, 0, 0])
        b = pp.array([0, 1, 0])
        result = pp.cross(a, b)
        self.assertEqual(result.tolist(), [0, 0, 1])

    def test_cross_anticommutative(self):
        a = pp.array([1, 2, 3])
        b = pp.array([4, 5, 6])
        ab = pp.cross(a, b)
        ba = pp.cross(b, a)
        result = ab + ba
        self.assertEqual(result.tolist(), [0, 0, 0])

    def test_cross_wrong_dimension(self):
        a = pp.array([1, 2])
        b = pp.array([3, 4])
        with self.assertRaises(InvalidOperationError):
            pp.cross(a, b)


class TestMatMul(unittest.TestCase):

    def test_identity_matmul(self):
        a = pp.array([[1, 2], [3, 4]])
        i = pp.identity(2)
        result = pp.matmul(a, i)
        self.assertEqual(result.tolist(), [[1, 2], [3, 4]])

    def test_basic_matmul(self):
        a = pp.array([[1, 2], [3, 4]])
        b = pp.array([[5, 6], [7, 8]])
        result = pp.matmul(a, b)
        self.assertEqual(result.tolist(), [[19, 22], [43, 50]])

    def test_non_square_matmul(self):
        a = pp.array([[1, 2, 3], [4, 5, 6]])
        b = pp.array([[7, 8], [9, 10], [11, 12]])
        result = pp.matmul(a, b)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[58, 64], [139, 154]])

    def test_matmul_shape_mismatch(self):
        a = pp.array([[1, 2], [3, 4]])
        b = pp.array([[1, 2, 3]])
        with self.assertRaises(ShapeMismatchError):
            pp.matmul(a, b)


class TestTranspose(unittest.TestCase):

    def test_square_transpose(self):
        a = pp.array([[1, 2], [3, 4]])
        result = pp.transpose(a)
        self.assertEqual(result.tolist(), [[1, 3], [2, 4]])

    def test_rectangular_transpose(self):
        a = pp.array([[1, 2, 3], [4, 5, 6]])
        result = pp.transpose(a)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_double_transpose(self):
        a = pp.array([[1, 2], [3, 4]])
        result = pp.transpose(pp.transpose(a))
        self.assertEqual(result.tolist(), a.tolist())


class TestMagnitude(unittest.TestCase):

    def test_unit_vector(self):
        a = pp.array([1, 0, 0])
        self.assertAlmostEqual(pp.magnitude(a), 1.0)

    def test_3d_magnitude(self):
        a = pp.array([3, 4, 0])
        self.assertAlmostEqual(pp.magnitude(a), 5.0)

    def test_zero_magnitude(self):
        a = pp.array([0, 0, 0])
        self.assertAlmostEqual(pp.magnitude(a), 0.0)


class TestNormalize(unittest.TestCase):

    def test_normalize_basic(self):
        a = pp.array([3, 4, 0])
        result = pp.normalize(a)
        expected = [0.6, 0.8, 0.0]
        for i in range(3):
            self.assertAlmostEqual(result.tolist()[i], expected[i])

    def test_normalize_unit_magnitude(self):
        a = pp.array([1, 2, 3])
        result = pp.normalize(a)
        self.assertAlmostEqual(pp.magnitude(result), 1.0)

    def test_normalize_zero_vector(self):
        a = pp.array([0, 0, 0])
        with self.assertRaises(InvalidOperationError):
            pp.normalize(a)


class TestDeterminant(unittest.TestCase):

    def test_1x1(self):
        a = pp.array([[5]])
        self.assertEqual(pp.determinant(a), 5)

    def test_2x2(self):
        a = pp.array([[1, 2], [3, 4]])
        self.assertAlmostEqual(pp.determinant(a), -2.0)

    def test_3x3(self):
        a = pp.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        self.assertAlmostEqual(pp.determinant(a), -306.0)

    def test_identity_det(self):
        i = pp.identity(3)
        self.assertAlmostEqual(pp.determinant(i), 1.0)

    def test_singular_det(self):
        a = pp.array([[1, 2], [2, 4]])
        self.assertAlmostEqual(pp.determinant(a), 0.0)


class TestInverse(unittest.TestCase):

    def test_2x2_inverse(self):
        a = pp.array([[4, 7], [2, 6]])
        inv = pp.inverse(a)
        product = pp.matmul(a, inv)
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(product.tolist()[i][j], expected, places=6)

    def test_3x3_inverse(self):
        a = pp.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        inv = pp.inverse(a)
        product = pp.matmul(a, inv)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(product.tolist()[i][j], expected, places=6)

    def test_singular_matrix_raises(self):
        a = pp.array([[1, 2], [2, 4]])
        with self.assertRaises(SingularMatrixError):
            pp.inverse(a)

    def test_1x1_inverse(self):
        a = pp.array([[4]])
        inv = pp.inverse(a)
        self.assertAlmostEqual(inv.tolist()[0][0], 0.25)


class TestTrace(unittest.TestCase):

    def test_trace_basic(self):
        a = pp.array([[1, 2], [3, 4]])
        self.assertEqual(pp.trace(a), 5)

    def test_trace_identity(self):
        i = pp.identity(4)
        self.assertEqual(pp.trace(i), 4)


class TestIdentity(unittest.TestCase):

    def test_identity_3x3(self):
        i = pp.identity(3)
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(i.tolist(), expected)
        self.assertEqual(i.shape, (3, 3))

    def test_identity_invalid(self):
        with self.assertRaises(InvalidOperationError):
            pp.identity(0)
        with self.assertRaises(InvalidOperationError):
            pp.identity(-1)


class TestZeros(unittest.TestCase):

    def test_zeros_vector(self):
        z = pp.zeros((3,))
        self.assertEqual(z.tolist(), [0, 0, 0])

    def test_zeros_matrix(self):
        z = pp.zeros((2, 3))
        self.assertEqual(z.tolist(), [[0, 0, 0], [0, 0, 0]])

    def test_zeros_3d(self):
        z = pp.zeros((2, 2, 2))
        self.assertEqual(z.shape, (2, 2, 2))


class TestDiagonal(unittest.TestCase):

    def test_diagonal_basic(self):
        a = pp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        d = pp.diagonal(a)
        self.assertEqual(d.tolist(), [1, 5, 9])

    def test_diagonal_non_square(self):
        a = pp.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(InvalidOperationError):
            pp.diagonal(a)


if __name__ == "__main__":
    unittest.main()
