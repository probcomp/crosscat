import crosscat.tests.quality_tests.quality_test_utils as qtu

import numpy

import unittest

def main():
    unittest.main()

class TestBincount(unittest.TestCase):
	def test_X_not_list_should_raise_exception(self):
		X = dict()
		self.assertRaises(TypeError, qtu.bincount, X)

		X = 2
		self.assertRaises(TypeError, qtu.bincount, X)

	def test_X_not_vector_should_raise_exception(self):
		X = numpy.zeros((2,2))
		self.assertRaises(ValueError, qtu.bincount, X)

	def test_bins_not_list_should_raise_exception(self):
		X = range(10)
		bins = dict()
		self.assertRaises(TypeError, qtu.bincount, X, bins=bins)

		bins = 12
		self.assertRaises(TypeError, qtu.bincount, X, bins=bins)

		bins = numpy.zeros(10)
		self.assertRaises(TypeError, qtu.bincount, X, bins=bins)

	def test_behavior_X_list(self):
		X = [0, 1, 2, 3]
		counts = qtu.bincount(X)
		assert counts == [1, 1, 1, 1]

		X = [1, 2, 2, 4, 6]
		counts = qtu.bincount(X)
		assert counts == [1, 2, 0, 1, 0, 1]

		bins = range(7)
		counts = qtu.bincount(X,bins)
		assert counts == [0, 1, 2, 0, 1, 0, 1]

		bins = [1,2,4,6]
		counts = qtu.bincount(X,bins)
		assert counts == [1, 2, 1, 1]

	def test_behavior_X_array(self):
		X = numpy.array([0, 1, 2, 3])
		counts = qtu.bincount(X)
		assert counts == [1, 1, 1, 1]

		X = numpy.array([1, 2, 2, 4, 6])
		counts = qtu.bincount(X)
		assert counts == [1, 2, 0, 1, 0, 1]

		bins = range(7)
		counts = qtu.bincount(X,bins)
		assert counts == [0, 1, 2, 0, 1, 0, 1]

		bins = [1,2,4,6]
		counts = qtu.bincount(X,bins)
		assert counts == [1, 2, 1, 1]

if __name__ == '__main__':
    main()