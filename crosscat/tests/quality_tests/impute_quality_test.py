#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
#   Research Leads: Vikash Mansinghka, Patrick Shafto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
import crosscat.cython_code.State as State
import crosscat.utils.sample_utils as su
import crosscat.utils.data_utils as du

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.quality_tests.synthetic_data_generator as sdg

import crosscat.tests.quality_tests.quality_test_utils as qtu

import random
import pylab
import numpy
from scipy import stats

import unittest

import crosscat.MultiprocessingEngine as mpe
import multiprocessing

distargs = dict(
	multinomial=dict(K=5),
	continuous=None,
	)

is_categorical = dict(
    multinomial=True,
    continuous=False,
    binomial=True,
    )


def main():
	print " "
	print "======================================================================="
	print "TEST IMPUTE VS. COLUMN AVERAGE"
	print " Checks that the mean error for impute vs the true data is less than"
	print " that of a column average."
	print " "
	print " ** NOTE: This test is not to be used for categorical data"
	print " "
	unittest.main()

class TestComponentModelQuality(unittest.TestCase):
	def test_normal_inverse_gamma_model_single(self):
		mse_sample, mse_ave = test_impute_vs_column_average_single(
								ccmext.p_ContinuousComponentModel)
		assert mse_sample < mse_ave

	def test_normal_inverse_gamma_model_multi(self):
		mse_sample, mse_ave = test_impute_vs_column_average_multi(
								ccmext.p_ContinuousComponentModel)
		assert mse_sample < mse_ave

def test_impute_vs_column_average_single(component_model_type, num_clusters=2,
				num_rows=100, separation=.9, seed=0, print_out=True):
	"""	tests predictive row generation vs column average
		Note: This test does not make sense for categorical data
		Inputs:
			- component_model_type: main class from datatype. Ex:
				ccmext.p_ContinuousComponentModel 
			- num_clusters: (optional) the number of clusters in the data
			- num_rows: (optional) the number of rows
			- separation: (optional) how well separated the clusters in each view are
				0 is idential distributions, 1 is well-separated
			- seed: (optional) int to seed the RNG 
			- print_out: (optional) bool, prints test information if True
		Returns:
			- the mean square error of the predictive sample column
			- the mean square error of the column average column
	"""

	if is_categorical[component_model_type.cctype]:
		raise TypeError("impute vs column average test doesn't make\
		 sense for categorical data types")

	random.seed(seed)
	get_next_seed = lambda : random.randrange(2147483647)

	N = num_rows
	C = separation
	
	# one sample for each processor
	num_samples = multiprocessing.cpu_count()

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

	# create a crosscat state 
	M_c = du.gen_M_c_from_T(T_list, cctypes=[cctype])
	M_r = du.gen_M_r_from_T(T_list)

	mstate = mpe.MultiprocessingEngine(cpu_count=num_samples)
	X_L_list, X_D_list = mstate.initialize(M_c, M_r, T_list, n_chains=num_samples)

	# transitions
	n_transitions=200
	X_L_list, X_D_list = mstate.analyze(M_c, T_list, X_L_list, X_D_list,
                            n_steps=n_transitions)

	column_average_errors = []
	crosscat_impute_errors = []

	teststr = "impute vs col average test (%s)" % component_model_type.cctype

	for chain in range(num_samples):

		qtu.print_progress(chain, num_samples, teststr)

		X_L = X_L_list[chain]
		X_D = X_D_list[chain]

		# generate a column entirely of imputed data
		T_generated = sdg.predictive_columns(M_c, X_L, X_D, [0],
				seed=get_next_seed(), impute=True)

		# generate a row of column averages
		T_colave = numpy.ones(T.shape)*numpy.mean(T)

		# get the mean squared error
		err_sample = numpy.mean( (T_generated-T)**2.0 )
		err_colave = numpy.mean( (T_colave-T)**2.0 )

		column_average_errors.append(err_colave)
		crosscat_impute_errors.append(err_sample)

	if print_out:
		print " "
		print "======================================"
		print "IMPUTE VS COLUMN AVERAGE (SINGLE COLUMN)"
		print "TEST INFORMATION:"
		print "       data type: " + component_model_type.cctype
		print "        num_rows: " + str(num_rows)
		print "    num_clusters: " + str(num_clusters)
		print "     num_samples: " + str(num_samples)
		print " num_transitions: " + str(n_transitions)
		print "RESULTS:"
		print "       crosscat mean error: " + str(["%.4f" % err for err in crosscat_impute_errors])
		print " column average mean error: " + str(["%.4f" % err for err in column_average_errors])

	del mstate
	return numpy.mean(crosscat_impute_errors), numpy.mean(column_average_errors)

def test_impute_vs_column_average_multi(component_model_type, 
	num_clusters=2, num_views=2, num_cols=4, num_rows=500, separation=.9, seed=0, print_out=True):
	"""	tests predictive row generation vs column average
		Note: This test does not make sense for categorical data
		Inputs:
			- component_model_type: main class from datatype. Ex:
				ccmext.p_ContinuousComponentModel 
			- num_clusters: (optional) the number of clusters in the data
			- num_views: (optional) the number of views
			- num_rows: (optional) the number of rows
			- separation: (optinal) how well separated the clusters in each view are
				0 is idential distributions, 1 is well-separated
			- seed: (optional) int to seed the RNG 
			- print_out: (optional) bool, prints test information if True
		Returns:
			- the mean square error of the predictive sample column
			- the mean square error of the column average column
	"""

	if is_categorical[component_model_type.cctype]:
		raise TypeError("impute vs column average test doesn't make\
		 sense for categorical data types")

	random.seed(seed)
	get_next_seed = lambda : random.randrange(2147483647)

	N = num_rows
	C = [separation]*num_views

	# one sample for each processor
	num_samples = multiprocessing.cpu_count()

	# generte a partition of columns to views (evenly-weighted)
	Z = range(num_views)
	for z in range(num_cols-num_views):
		Z.append(random.randrange(num_views))

	random.shuffle(Z)

	# data types from rows
	cctypes = [component_model_type.cctype]*num_cols

	# extra data type arguments for SDG
	dtargs = [distargs[component_model_type.cctype]]*num_cols

	cluster_weights = [[1.0/float(num_clusters)]*num_clusters]*num_views

	T_list, M_c = sdg.gen_data(cctypes, num_rows, Z, cluster_weights, C,
				seed=get_next_seed(), distargs=dtargs)

	T = numpy.array(T_list)

	# create a crosscat state 
	M_r = du.gen_M_r_from_T(T_list)

	mstate = mpe.MultiprocessingEngine(cpu_count=num_samples)
	X_L_list, X_D_list = mstate.initialize(M_c, M_r, T_list, n_chains=num_samples)

	# transitions
	n_transitions=200
	X_L_list, X_D_list = mstate.analyze(M_c, T_list, X_L_list, X_D_list,
                            n_steps=n_transitions)

	column_average_errors = []
	crosscat_impute_errors = []

	teststr = "impute vs col average test (%s)" % component_model_type.cctype

	for chain in range(num_samples):

		qtu.print_progress(chain, num_samples, teststr)

		X_L = X_L_list[chain]
		X_D = X_D_list[chain]

		# generate a columns from the sample
		T_generated = sdg.predictive_columns(M_c, X_L, X_D, range(num_cols),
			seed=get_next_seed(), impute=True)

		# generate a row of column averages
		T_colave = numpy.tile(numpy.mean(T, axis=0), (num_rows,1))

		# get the mean squared error for the entire matrix
		err_sample = numpy.mean( (T_generated-T)**2.0 )
		err_colave = numpy.mean( (T_colave-T)**2.0 )

		column_average_errors.append(err_colave)
		crosscat_impute_errors.append(err_sample)

	if print_out:
		print " "
		print "======================================"
		print "IMPUTE VS COLUMN AVERAGE (MULTICOLUMN)"
		print "TEST INFORMATION:"
		print "       data type: " + component_model_type.cctype
		print "        num_rows: " + str(num_rows)
		print "        num_cols: " + str(num_cols)
		print "       num_views: " + str(num_views)
		print "    num_clusters: " + str(num_clusters)
		print "		    Z_views: " + str(Z)
		print "     num_samples: " + str(num_samples)
		print " num_transitions: " + str(n_transitions)
		print "RESULTS:"
		print "       crosscat mean error: " + str(["%.4f" % err for err in crosscat_impute_errors])
		print " column average mean error: " + str(["%.4f" % err for err in column_average_errors])

	del mstate
	return numpy.mean(crosscat_impute_errors), numpy.mean(column_average_errors)


if __name__ == '__main__':
    main()