"""Unit tests for PotatoNumPy tensor operations."""

import unittest

import potatonumpy as pp
from potatonumpy.exceptions import InvalidOperationError


class TestTensorSum(unittest.TestCase):

    def test_sum_vector(self):
        a = pp.array([1, 2, 3, 4])
        self.assertEqual(pp.tensor_sum(a), 10)

    def test_sum_matrix(self):
        a = pp.array([[1, 2], [3, 4]])
        self.assertEqual(pp.tensor_sum(a), 10)

    def test_sum_3d(self):
        a = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(pp.tensor_sum(a), 36)

    def test_sum_axis_0(self):
        a = pp.array([[1, 2], [3, 4]])
        result = pp.tensor_sum(a, axis=0)
        self.assertEqual(result.tolist(), [4, 6])

    def test_sum_axis_1(self):
        a = pp.array([[1, 2], [3, 4]])
        result = pp.tensor_sum(a, axis=1)
        self.assertEqual(result.tolist(), [3, 7])

    def test_sum_invalid_axis(self):
        a = pp.array([1, 2, 3])
        with self.assertRaises(InvalidOperationError):
            pp.tensor_sum(a, axis=2)


class TestTensorMin(unittest.TestCase):

    def test_min_vector(self):
        a = pp.array([3, 1, 4, 1, 5])
        self.assertEqual(pp.tensor_min(a), 1)

    def test_min_matrix(self):
        a = pp.array([[10, 2], [8, 4]])
        self.assertEqual(pp.tensor_min(a), 2)

    def test_min_axis_0(self):
        a = pp.array([[3, 1], [2, 4]])
        result = pp.tensor_min(a, axis=0)
        self.assertEqual(result.tolist(), [2, 1])

    def test_min_axis_1(self):
        a = pp.array([[3, 1], [2, 4]])
        result = pp.tensor_min(a, axis=1)
        self.assertEqual(result.tolist(), [1, 2])


class TestTensorMax(unittest.TestCase):

    def test_max_vector(self):
        a = pp.array([3, 1, 4, 1, 5])
        self.assertEqual(pp.tensor_max(a), 5)

    def test_max_matrix(self):
        a = pp.array([[10, 2], [8, 14]])
        self.assertEqual(pp.tensor_max(a), 14)

    def test_max_axis_0(self):
        a = pp.array([[3, 1], [2, 4]])
        result = pp.tensor_max(a, axis=0)
        self.assertEqual(result.tolist(), [3, 4])


class TestTensorMean(unittest.TestCase):

    def test_mean_vector(self):
        a = pp.array([2, 4, 6, 8])
        self.assertAlmostEqual(pp.tensor_mean(a), 5.0)

    def test_mean_matrix(self):
        a = pp.array([[1, 2], [3, 4]])
        self.assertAlmostEqual(pp.tensor_mean(a), 2.5)


class TestTensorOperations3D(unittest.TestCase):

    def test_3d_elementwise_add(self):
        a = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = pp.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        result = a + b
        self.assertEqual(result.shape, (2, 2, 2))
        expected = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]]
        self.assertEqual(result.tolist(), expected)

    def test_3d_scalar_broadcast(self):
        a = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = a * 2
        self.assertEqual(result.shape, (2, 2, 2))
        expected = [[[2, 4], [6, 8]], [[10, 12], [14, 16]]]
        self.assertEqual(result.tolist(), expected)

    def test_3d_sum_axis_0(self):
        a = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = pp.tensor_sum(a, axis=0)
        self.assertEqual(result.tolist(), [[6, 8], [10, 12]])

    def test_3d_sum_axis_1(self):
        a = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = pp.tensor_sum(a, axis=1)
        self.assertEqual(result.tolist(), [[4, 6], [12, 14]])

    def test_3d_sum_axis_2(self):
        a = pp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = pp.tensor_sum(a, axis=2)
        self.assertEqual(result.tolist(), [[3, 7], [11, 15]])


if __name__ == "__main__":
    unittest.main()
