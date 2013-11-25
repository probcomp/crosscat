import crosscat.tests.quality_tests.synthetic_data_generator as sdg

import unittest

def main():
    unittest.main()

class TestSyntheticDataGenerator(unittest.TestCase):
	def setUp(self):
		self.cctypes_all_contiuous = ['continuous']*5
		self.cctypes_all_multinomial = ['multinomial']*5
		self.cctypes_mixed = ['continuous','continuous','multinomial','continuous','multinomial']
		self.cctypes_wrong_type = dict()

		self.n_rows = 10;

		self.cols_to_views_good = [0, 0, 1, 2, 1]
		self.cols_to_views_bad_start_index = [3, 3, 1, 2, 1]
		self.cols_to_views_skip_value = [0, 0, 1, 3, 1]
		self.cols_to_views_wrong_type = dict()

		self.cluster_weights_good = [[.2, .2, .6],[.5, .5],[.8, .2]]
		self.cluster_weights_missing_view = [[.2, .2, .6],[.8, .2]]
		self.cluster_weights_bad_sum = [[.2, .2, .6],[.1, .5],[.8, .2]]
		self.cluster_weights_wrong_type = dict()

		self.separation_good = [.4, .5, .9];
		self.separation_out_of_range_low = [-1, .5, .9];
		self.separation_out_of_range_high = [1.5, .5, .9];
		self.separation_wrong_number_views = [.5, .9];
		self.separation_wrong_type = dict();


	def test_proper_set_up_all_continuous(self):
		T, M_c = sdg.gen_data(self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		assert(len(T) == self.n_rows)
		assert(len(T[0]) == len(self.cols_to_views_good))

	def test_proper_set_up_all_multinomial(self):
		distargs = [dict(K=5), dict(K=5), dict(K=5), dict(K=5), dict(K=5)]
		T, M_c = sdg.gen_data(self.cctypes_all_multinomial,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=distargs)

		assert(len(T) == self.n_rows)
		assert(len(T[0]) == len(self.cols_to_views_good))

	def test_proper_set_up_mixed(self):
		# cctypes = ['continuous','continuous','multinomial','continuous','multinomial']
		distargs = [ None, None, dict(K=5), None, dict(K=5)]
		T, M_c = sdg.gen_data(self.cctypes_mixed,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=distargs)

		assert(len(T) == self.n_rows)
		assert(len(T[0]) == len(self.cols_to_views_good))

	def test_bad_cctypes_should_raise_exception(self):
		# wrong type (dict)
		self.assertRaises(TypeError, sdg.gen_data,
			dict(),
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		# empty list
		self.assertRaises(ValueError, sdg.gen_data,
			[],
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		# invalid cctype (peanut)
		self.assertRaises(ValueError, sdg.gen_data,
			['continuous','continuous','continuous','continuous','peanut'],
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		# number of columns too low (should be 5)
		self.assertRaises(ValueError, sdg.gen_data,
			['continuous']*4,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		# number of columns too high (should be 5)
		self.assertRaises(ValueError, sdg.gen_data,
			['continuous']*6,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

	def test_bad_cols_to_views_should_raise_exception(self):
		# start index with 1 instead of 0
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_bad_start_index,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		# skip indices
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_skip_value,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

		# give a dict instead of a list
		self.assertRaises(TypeError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_wrong_type,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=None)

	def test_bad_cluster_weights_should_raise_exception(self):
		# number of views is too low
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_missing_view,
			self.separation_good,
			seed=0,
			distargs=None)

		# cluster weights do not sum to 1
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_bad_sum,
			self.separation_good,
			seed=0,
			distargs=None)

		# dict instead of list of lists
		self.assertRaises(TypeError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_wrong_type,
			self.separation_good,
			seed=0,
			distargs=None)

	def test_bad_separation_should_raise_exception(self):
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_out_of_range_low,
			seed=0,
			distargs=None)

		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_out_of_range_high,
			seed=0,
			distargs=None)

		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_wrong_number_views,
			seed=0,
			distargs=None)

		self.assertRaises(TypeError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_wrong_type,
			seed=0,
			distargs=None)

	def test_bad_distargs_should_raise_exception(self):
		# wrong type
		self.assertRaises(TypeError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=10)

		# wrong number of entries
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=[None]*4)

		# wrong entry type
		self.assertRaises(ValueError, sdg.gen_data,
			self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=[dict(K=5)]*5)

		# wrong entry type
		self.assertRaises(TypeError, sdg.gen_data,
			self.cctypes_all_multinomial,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=[None]*5)

		# wrong dict entry for multinomial
		self.assertRaises(KeyError, sdg.gen_data,
			self.cctypes_all_multinomial,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=[dict(P=12)]*5)


if __name__ == '__main__':
    main()