import crosscat.tests.synthetic_data_generator as sdg
import crosscat.cython_code.State as State
import crosscat.utils.data_utils as du

import unittest
import random
import numpy

def main():
    unittest.main()


class TestPredictiveColumns(unittest.TestCase):
	def setUp(self):
		# generate a crosscat state and pull the metadata
		gen_seed = 0
		num_clusters = 2
		self.num_rows = 10
		self.num_cols = 2
		num_splits = 1

		self.T, self.M_r, self.M_c = du.gen_factorial_data_objects(gen_seed,
							   num_clusters, self.num_cols, 
							   self.num_rows, num_splits)

		state = State.p_State(self.M_c, self.T)
		self.X_L = state.get_X_L()
		self.X_D = state.get_X_D()

	def test_should_return_array_of_proper_size(self):
		columns_list = [0]
		X = sdg.predictive_columns(self.M_c, self.X_L, self.X_D, columns_list)
		assert isinstance(X, numpy.ndarray)
		assert X.shape[0] == self.num_rows
		assert X.shape[1] == len(columns_list)

		columns_list = [0,1]
		X = sdg.predictive_columns(self.M_c, self.X_L, self.X_D, columns_list)
		assert isinstance(X, numpy.ndarray)
		assert X.shape[0] == self.num_rows
		assert X.shape[1] == len(columns_list)

	def test_should_not_generate_data_from_invalid_rows(self):
		columns_list = [0,-1]
		self.assertRaises(ValueError, sdg.predictive_columns, 
			self.M_c, self.X_L, self.X_D, columns_list)

		columns_list = [0,3]
		self.assertRaises(ValueError, sdg.predictive_columns, 
			self.M_c, self.X_L, self.X_D, columns_list)

	def test_should_have_nan_entries_if_specified(self):
		# for one column
		columns_list = [0]
		optargs = [dict(missing_data=1.0)] # every entry will be missing NaN
		X = sdg.predictive_columns(self.M_c, self.X_L, self.X_D, columns_list,
			optional_settings=optargs)

		assert numpy.all(numpy.isnan(X))
		
		# for two columns
		columns_list = [0,1]
		optargs = [dict(missing_data=1.0)]*2 
		X = sdg.predictive_columns(self.M_c, self.X_L, self.X_D, columns_list,
			optional_settings=optargs)

		assert numpy.all(numpy.isnan(X))

		# for one of two columns (no dict means no missing data)
		columns_list = [0,1]
		optargs = [dict(missing_data=1.0), None]
		X = sdg.predictive_columns(self.M_c, self.X_L, self.X_D, columns_list,
			optional_settings=optargs)

		assert numpy.all(numpy.isnan(X[:,0]))
		assert not numpy.any(numpy.isnan(X[:,1]))

		# for one of two columns. Missing data specified 0 for second column
		columns_list = [0,1]
		optargs = [dict(missing_data=1.0), dict(missing_data=0.0)]
		X = sdg.predictive_columns(self.M_c, self.X_L, self.X_D, columns_list,
			optional_settings=optargs)

		assert numpy.all(numpy.isnan(X[:,0]))
		assert not numpy.any(numpy.isnan(X[:,1]))

class TestGenerateGeparatedGodelParameters(unittest.TestCase):
	def setUp(self):
		self.num_clusters = 5
		self.get_next_seed = lambda : random.randrange(32000)
		self.distargs_multinomial = dict(K=5)
		random.seed(0)

	def test_should_return_list_of_params(self):
		ret = sdg.generate_separated_model_parameters('continuous',
			.5, self.num_clusters, self.get_next_seed )

		assert isinstance(ret, list)
		assert len(ret) == self.num_clusters
		for entry in ret:
			assert isinstance(entry, dict)
			for key in entry.keys():
				assert key in ['mu', 'rho']

			assert len(entry.keys()) == 2

		ret = sdg.generate_separated_model_parameters('multinomial',
			.5, self.num_clusters, self.get_next_seed,
			distargs=self.distargs_multinomial)

		assert isinstance(ret, list)
		assert len(ret) == self.num_clusters
		for entry in ret:
			assert isinstance(entry, dict)
			for key in entry.keys():
				assert key in ['weights']

			assert len(entry.keys()) == 1

	def tests_should_not_accept_invalid_cctype(self):
		# peanut is an invalid cctype
		self.assertRaises(ValueError, sdg.generate_separated_model_parameters,
			'peanut', .5, self.num_clusters, self.get_next_seed)

	def test_normal_means_should_be_farther_apart_if_they_have_higer_separation(self):
		random.seed(0)	
		closer = sdg.generate_separated_model_parameters('continuous',
			.1, 2, self.get_next_seed )

		sum_std_close = closer[0]['rho']**(-.5) + closer[1]['rho']**(-.5)
		distance_close = ((closer[0]['mu']-closer[1]['mu'])/sum_std_close)**2.0

		random.seed(0)
		farther = sdg.generate_separated_model_parameters('continuous',
			.5, 2, self.get_next_seed )

		sum_std_far = farther[0]['rho']**(-.5) + farther[1]['rho']**(-.5)
		distance_far = ((farther[0]['mu']-farther[1]['mu'])/sum_std_far)**2.0

		random.seed(0)
		farthest = sdg.generate_separated_model_parameters('continuous',
			1.0, 2, self.get_next_seed )

		sum_std_farthest = farthest[0]['rho']**(-.5) + farthest[1]['rho']**(-.5)
		distance_farthest = ((farthest[0]['mu']-farthest[1]['mu'])/sum_std_farthest)**2.0

		assert distance_far  > distance_close
		assert distance_farthest  > distance_far


class TestsGenerateSeparatedMultinomialWeights(unittest.TestCase):
	def setUp(self):
		self.A_good = [.2]*5
		self.C_good = .5

	def tests_should_return_proper_list(self):
		w = sdg.generate_separated_multinomial_weights(self.A_good,self.C_good)
		assert isinstance(w, list)
		assert len(w) == len(self.A_good)

	def tests_bad_separation_should_raise_exception(self):
		# C is too low
		self.assertRaises(ValueError, sdg.generate_separated_multinomial_weights,
			self.A_good, -.1)
		# C is too high
		self.assertRaises(ValueError, sdg.generate_separated_multinomial_weights,
			self.A_good, 1.2)

	def tests_bad_weights_should_raise_exception(self):
		# weights do not sum to 1
		self.assertRaises(ValueError, sdg.generate_separated_multinomial_weights,
			[.2]*4, .5)

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


	def test_same_seeds_should_produce_the_same_data(self):
		distargs = [None]*5
		T1, M_c = sdg.gen_data(self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=distargs)

		T2, M_c = sdg.gen_data(self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=distargs)

		A1 = numpy.array(T1)
		A2 = numpy.array(T2)

		assert numpy.all(A1==A2)

	def test_different_seeds_should_produce_the_different_data(self):
		distargs = [None]*5
		T1, M_c = sdg.gen_data(self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=0,
			distargs=distargs)

		T2, M_c = sdg.gen_data(self.cctypes_all_contiuous,
			self.n_rows,
			self.cols_to_views_good,
			self.cluster_weights_good,
			self.separation_good,
			seed=12345,
			distargs=distargs)

		A1 = numpy.array(T1)
		A2 = numpy.array(T2)
		
		assert not numpy.all(A1==A2)

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