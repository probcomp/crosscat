import crosscat.cython_code.State as State
import crosscat.utils.sample_utils as su
import crosscat.utils.data_utils as du

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext
import crosscat.tests.synthetic_data_generator as sdg

import crosscat.tests.quality_test_utils as qtu

import random
import pylab
import numpy
from scipy import stats

import unittest

distargs = dict(
	multinomial=dict(K=5),
	continuous=None,
	)

def main():
    unittest.main()

class TestComponentModelQuality(unittest.TestCase):
    def test_normal_inverse_gamma_model(self):
    	mse_sample, mse_ave = check_impute_vs_column_average_single(
    							ccmext.p_ContinuousComponentModel, 2)
        assert mse_sample < mse_ave

def check_impute_vs_column_average_single(component_model_type, num_clusters, seed=0):
	"""	tests predictive row generation vs column average
		Note: This test does not make sense for categorical data
		Inputs:
			- component_model_type: main class from datatype. Ex:
				ccmext.p_ContinuousComponentModel 
			- num_clusters: the number of clusters in the data
			- seed: (optional) int to seed the RNG 
		Returns:
			- the mean square error of the predictive sample column
			- the mean square error of the column average column
	"""

	random.seed(seed)

	N = 100

	get_next_seed = lambda : random.randrange(2147483647)

	C = .9 # highly-separated clusters

	cctype = component_model_type.cctype

	component_model_parameters = sdg.generate_separated_model_parameters(
						cctype, C, num_clusters, get_next_seed,
						distargs=distargs[cctype])

	# generte a partition of rows to clusters (evenly-weighted)
	Z = range(num_clusters)
	for z in range(N-num_clusters):
		Z.append(random.randrange(num_clusters))

	random.shuffle(Z)

	# generate the data
	T = numpy.array([[0]]*N, dtype=float)

	for x in range(N):
		z = Z[x]
		T[x] = component_model_type.generate_data_from_parameters(
				component_model_parameters[z], 1, gen_seed=get_next_seed())[0]

	T_list = T.tolist()

	# intialize the state
	M_c = du.gen_M_c_from_T(T_list, cctypes=[cctype])

	state = State.p_State(M_c, T)

	# transitions
	state.transition(n_steps=100)

	# get the sample
	X_L = state.get_X_L()
	X_D = state.get_X_D()

	# generate a row from the sample
	T_generated = sdg.predictive_columns(M_c, X_L, X_D, [0], seed=get_next_seed())

	# generate a row of column averages
	T_colave = numpy.ones(T.shape)*numpy.mean(T)

	# get the mean squared error
	err_sample = numpy.mean( (T_generated-T)**2.0 )
	err_colave = numpy.mean( (T_colave-T)**2.0 )

	return err_sample, err_colave

if __name__ == '__main__':
    main()