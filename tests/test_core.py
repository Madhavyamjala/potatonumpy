"""Unit tests for PotatoNumPy core array operations."""

import unittest

import potatonumpy as pp
from potatonumpy.exceptions import (
    ShapeMismatchError,
    InvalidOperationError,
    InvalidTensorError,
)


class TestArrayCreation(unittest.TestCase):

    def test_create_from_list(self):
        a = pp.array([1, 2, 3])
        self.assertEqual(a.shape, (3,))
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.size, 3)

    def test_create_from_nested_list(self):
        a = pp.array([[1, 2], [3, 4]])
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.size, 4)

    def test_create_from_scalar(self):
        a = pp.array(42)
        self.assertEqual(a.shape, ())
        self.assertEqual(a.ndim, 0)
        self.assertEqual(a.size, 1)

    def test_create_3d_tensor(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        a = pp.array(data)
        self.assertEqual(a.shape, (2, 2, 2))
        self.assertEqual(a.ndim, 3)
        self.assertEqual(a.size, 8)

    def test_create_with_floats(self):
        a = pp.array([1.5, 2.5, 3.5])
        self.assertEqual(a.tolist(), [1.5, 2.5, 3.5])

    def test_create_with_complex(self):
        a = pp.array([1 + 2j, 3 + 4j])
        self.assertEqual(a.tolist(), [1 + 2j, 3 + 4j])

    def test_deep_copy(self):
        original = [[1, 2], [3, 4]]
        a = pp.array(original)
        original[0][0] = 999
        self.assertEqual(a.data[0][0], 1)


class TestInvalidArrays(unittest.TestCase):

    def test_ragged_array(self):
        with self.assertRaises(InvalidTensorError):
            pp.array([[1, 2], [3, 4, 5]])

    def test_mixed_types_at_depth(self):
        with self.assertRaises(InvalidTensorError):
            pp.array([[1, 2], 3])

    def test_empty_array(self):
        with self.assertRaises(InvalidTensorError):
            pp.array([])

    def test_invalid_type(self):
        with self.assertRaises(InvalidTensorError):
            pp.array("hello")

    def test_invalid_element_type(self):
        with self.assertRaises(InvalidTensorError):
            pp.array([1, "two", 3])


class TestElementwiseOps(unittest.TestCase):

    def test_add_vectors(self):
        a = pp.array([1, 2, 3])
        b = pp.array([4, 5, 6])
        result = a + b
        self.assertEqual(result.tolist(), [5, 7, 9])

    def test_sub_vectors(self):
        a = pp.array([10, 20, 30])
        b = pp.array([1, 2, 3])
        result = a - b
        self.assertEqual(result.tolist(), [9, 18, 27])

    def test_mul_vectors(self):
        a = pp.array([2, 3, 4])
        b = pp.array([5, 6, 7])
        result = a * b
        self.assertEqual(result.tolist(), [10, 18, 28])

    def test_div_vectors(self):
        a = pp.array([10.0, 20.0, 30.0])
        b = pp.array([2.0, 4.0, 5.0])
        result = a / b
        self.assertEqual(result.tolist(), [5.0, 5.0, 6.0])

    def test_pow_vectors(self):
        a = pp.array([2, 3, 4])
        result = a ** 2
        self.assertEqual(result.tolist(), [4, 9, 16])

    def test_mod_vectors(self):
        a = pp.array([10, 11, 12])
        result = a % 3
        self.assertEqual(result.tolist(), [1, 2, 0])

    def test_add_matrices(self):
        a = pp.array([[1, 2], [3, 4]])
        b = pp.array([[5, 6], [7, 8]])
        result = a + b
        self.assertEqual(result.tolist(), [[6, 8], [10, 12]])

    def test_scalar_broadcast_add(self):
        a = pp.array([1, 2, 3])
        result = a + 10
        self.assertEqual(result.tolist(), [11, 12, 13])

    def test_scalar_broadcast_radd(self):
        a = pp.array([1, 2, 3])
        result = 10 + a
        self.assertEqual(result.tolist(), [11, 12, 13])

    def test_scalar_broadcast_rsub(self):
        a = pp.array([1, 2, 3])
        result = 10 - a
        self.assertEqual(result.tolist(), [9, 8, 7])

    def test_scalar_broadcast_rmul(self):
        a = pp.array([1, 2, 3])
        result = 2 * a
        self.assertEqual(result.tolist(), [2, 4, 6])

    def test_neg(self):
        a = pp.array([1, -2, 3])
        result = -a
        self.assertEqual(result.tolist(), [-1, 2, -3])

    def test_abs(self):
        a = pp.array([-1, -2, 3])
        result = abs(a)
        self.assertEqual(result.tolist(), [1, 2, 3])


class TestElementwiseErrors(unittest.TestCase):

    def test_shape_mismatch_add(self):
        a = pp.array([1, 2, 3])
        b = pp.array([1, 2])
        with self.assertRaises(ShapeMismatchError):
            a + b

    def test_division_by_zero(self):
        a = pp.array([1, 2, 3])
        with self.assertRaises(InvalidOperationError):
            a / 0

    def test_mod_by_zero(self):
        a = pp.array([1, 2, 3])
        with self.assertRaises(InvalidOperationError):
            a % 0

    def test_invalid_operand_type(self):
        a = pp.array([1, 2, 3])
        with self.assertRaises(InvalidOperationError):
            a + "hello"


class TestIndexing(unittest.TestCase):

    def test_index_vector(self):
        a = pp.array([10, 20, 30])
        self.assertEqual(a[0], 10)
        self.assertEqual(a[2], 30)

    def test_index_matrix_row(self):
        a = pp.array([[1, 2], [3, 4]])
        row = a[0]
        self.assertEqual(row.tolist(), [1, 2])

    def test_index_scalar_raises(self):
        a = pp.array(42)
        with self.assertRaises(InvalidOperationError):
            a[0]

    def test_len_vector(self):
        a = pp.array([1, 2, 3, 4, 5])
        self.assertEqual(len(a), 5)

    def test_len_scalar_raises(self):
        a = pp.array(42)
        with self.assertRaises(InvalidOperationError):
            len(a)


class TestFlattenReshape(unittest.TestCase):

    def test_flatten(self):
        a = pp.array([[1, 2], [3, 4]])
        flat = a.flatten()
        self.assertEqual(flat.tolist(), [1, 2, 3, 4])
        self.assertEqual(flat.shape, (4,))

    def test_reshape(self):
        a = pp.array([1, 2, 3, 4, 5, 6])
        reshaped = a.reshape((2, 3))
        self.assertEqual(reshaped.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(reshaped.shape, (2, 3))

    def test_reshape_invalid_size(self):
        a = pp.array([1, 2, 3, 4, 5])
        with self.assertRaises(InvalidTensorError):
            a.reshape((2, 3))


class TestStringRepresentation(unittest.TestCase):

    def test_str_vector(self):
        a = pp.array([1, 2, 3])
        s = str(a)
        self.assertIn("PotatoNumPy", s)
        self.assertIn("1", s)

    def test_repr(self):
        a = pp.array([1, 2])
        r = repr(a)
        self.assertIn("shape", r)
        self.assertIn("data", r)


class TestEquality(unittest.TestCase):

    def test_equal_arrays(self):
        a = pp.array([1, 2, 3])
        b = pp.array([1, 2, 3])
        self.assertTrue(a == b)

    def test_unequal_arrays(self):
        a = pp.array([1, 2, 3])
        b = pp.array([1, 2, 4])
        self.assertFalse(a == b)

    def test_different_shapes(self):
        a = pp.array([1, 2])
        b = pp.array([1, 2, 3])
        self.assertFalse(a == b)


if __name__ == "__main__":
    unittest.main()
