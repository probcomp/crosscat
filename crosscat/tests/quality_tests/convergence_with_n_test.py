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
import crosscat.utils.convergence_test_utils as ctu

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext
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
	multinomial=dict(K=8),
	continuous=None,
	)

is_discrete = dict(
    multinomial=True,
    continuous=False
    )

def main():
    unittest.main()

class TestImprovementWithN(unittest.TestCase):
	def setUp(self):
		print " "
		print "======================================================================="
		print "TEST KL-DIVERGENCE GOES TO 0 WITH N"
		print " Checks that ARI increases and KL-divergence decreases as a function"
		print " of the number of rows."
		print " "
		print " **NOTE: This will take a while to run"
		self.show_plot=True

	# FIXME: The tests below are commented out because there is a problem (at
	# least on OSX) in which pools are not (whether or not del is used) 
	# closed until unittest is done. This makes my computer cross.
	def test_normal_inverse_gamma_improves_with_N_multi(self):
		improvement = multi_column_convergence_test(ccmext.p_ContinuousComponentModel,
						num_cols=8,
						seed=0, show_plot=self.show_plot)

		assert improvement

	# def test_normal_inverse_gamma_improves_with_N_single(self):
	# 	improvement = multi_column_convergence_test(ccmext.p_ContinuousComponentModel,
	# 					num_cols=1, num_views=1,
	# 					seed=random.randrange(32000), show_plot=self.show_plot)

	# 	assert improvement

	# def test_dirichlet_multinomial_improves_with_N_single(self):
	# 	improvement = multi_column_convergence_test(mcmext.p_MultinomialComponentModel,
	# 					num_cols=1, num_views=1,
	# 					seed=random.randrange(32000), show_plot=self.show_plot)

	# 	assert improvement

	# def test_dirichlet_multinomial_improves_with_N_multi(self):
	# 	improvement = multi_column_convergence_test(mcmext.p_MultinomialComponentModel,
	# 					num_cols=8,
	# 					seed=0, show_plot=self.show_plot)

	# 	assert improvement
		

def multi_column_convergence_test(component_model_type, num_cols=4, num_views=2,
	num_clusters=2, separation=.9,
	num_chains=None, num_transition=500, N_list=None, seed=0,
	show_plot=True, print_out=True):

	if num_chains is None:
		num_chains = multiprocessing.cpu_count()


	random.seed(seed)
	numpy.random.seed(seed)
	get_next_seed = lambda : random.randrange(2147483647)

	# generate a partition of cols to views
	Z_v = range(num_views)
	for _ in range(num_cols-num_views):
		Z_v.append(random.randrange(num_views))

	weights = [[1.0/float(num_clusters)]*num_clusters]*num_views
	
	cctype = component_model_type.cctype

	# generates a distribution (we just want the structure, the number of rows
	# is irrelevant)
	T, M_c, struc = sdg.gen_data([cctype]*num_cols, 100, Z_v, weights, 
				[separation]*num_views, 
				seed=get_next_seed(), distargs=[distargs[cctype]]*num_cols, 
				return_structure=True)

	support = qtu.get_mixture_support(cctype, component_model_type, 
					struc['component_params'][0], nbins=1000, support=.95)

	# number of data point to draw
	if N_list is None:
		N_list = [10, 18, 32, 58, 105, 190, 342, 616, 1110, 2000]

	KL = numpy.zeros((num_cols, len(N_list)))
	ARI = numpy.zeros((num_cols, len(N_list)))

	n = 0
	for N in N_list:
		teststr = "Improvement with N (n_col=%i, N=%i, %s)" % (num_cols, N, cctype)
		qtu.print_progress(n, len(N_list), teststr)
		T_n, Z = sdg.gen_crosscat_array_from_params(struc, N, return_partitions=True)
		struc['rows_to_clusters'] = Z
		kl = single_run(T_n, M_c, struc, component_model_type,
			 		num_chains, num_transition, seed)

		KL[:,n] = kl
		# ARI[:,n] = ari

		n += 1

	qtu.print_progress(n, len(N_list), teststr)

	if show_plot:
		ax = pylab.subplot(1,1,1)
		for c in range(num_cols):
			pylab.plot(N_list, KL[c,:], color="black", alpha=.5)
		pylab.plot(N_list, numpy.mean(KL, axis=0), color="red", linewidth=3)
		pylab.title('mean log pdf squared error')
		pylab.xlabel('N')
		ax.set_xscale('log')

		# TODO: using ARI
		# ax = pylab.subplot(1,2,2)
		# pylab.plot(N_list, numpy.mean(ARI, axis=0))
		# pylab.title('ARI')
		# pylab.xlabel('N')
		# ax.set_xscale('log')

		pylab.show()

	X = numpy.tile(numpy.array(N_list, dtype=float), (1,num_cols))

	# fit regression lines to the change over time
	lr_err = stats.linregress(X.flatten(1), KL.flatten(1))

	# KL divergence should decrease over time
	error_decreased = lr_err[0] < 0 and lr_err[3] < .05
	

	if print_out:
		print " "
		print "======================================"
		print "CONVERGENCE TO"
		print "TEST INFORMATION:"
		print "       data type: " + component_model_type.cctype
		print "        num_cols: " + str(num_cols)
		print "          N_list: " + str(N_list)
		print "    num_clusters: " + str(num_clusters)
		print "     num_samples: " + str(num_chains)
		print " num_transitions: " + str(num_transition)
		print "RESULTS (LOG PDF ERROR REGRESSION):"
		print "     slope: " + str(lr_err[0])
		print "   r-value: " + str(lr_err[2])
		print "   p-value: " + str(lr_err[3])
		print "    PASSED: " + str(error_decreased)

	return error_decreased

def single_run(T, M_c, structure, component_model_type,
	n_chains, n_transitions=100, seed=0):


	random.seed(seed)
	numpy.random.seed(seed)
	get_next_seed = lambda : random.randrange(2147483647)

	cctype = component_model_type.cctype

	M_r = du.gen_M_r_from_T(T)
	mstate = mpe.MultiprocessingEngine(cpu_count=multiprocessing.cpu_count())
	X_L_list, X_D_list = mstate.initialize(M_c, M_r, T, n_chains=n_chains)

	T_array = numpy.array(T)

	num_rows, num_cols = T_array.shape

	supports = []
	true_log_pdfs = []

	# pdb.set_trace()

	for c in range(num_cols):
		v = structure['cols_to_views'][c]
		support = qtu.get_mixture_support(cctype, component_model_type, 
						structure['component_params'][c], nbins=100, support=.999)
		true_log_pdf = qtu.get_mixture_pdf(support, component_model_type, 
						structure['component_params'][c], structure['cluster_weights'][v])

		supports.append(support)
		true_log_pdfs.append(true_log_pdf)

	# abs_error_P = numpy.zeros((num_cols, len(true_p)))
	X_err = numpy.zeros((num_cols, num_rows))
	KL = numpy.zeros((num_cols, n_chains))

	X_L_list, X_D_list = mstate.analyze(M_c, T, X_L_list, X_D_list, n_steps=n_transitions)

	# ARI
	ARI = ctu.multi_chain_ARI(X_L_list, X_D_list, structure['cols_to_views'], 
		structure['rows_to_clusters'])[0]

	teststr = "Improvement with N (single col, inner loop, N-%i, %s)" % (num_rows, cctype)

	for s in range(n_chains):

		# qtu.print_progress(s+1, n_chains, teststr)
		# get partitions and generate a predictive column
		X_L = X_L_list[s]
		X_D = X_D_list[s]
		for c in range(num_cols):

			support = supports[c]
			true_log_pdf = true_log_pdfs[c]

			# KL divergence
			v = structure['cols_to_views'][c]
			# NOTE: Removed KL divergence here because the estimation method I use
			# (quadrature) for continuous distributions some produces negative values
			# (which occurs when the support for the inferred and true distributions
			# are significantly different)
			# KL[c,s] = qtu.KL_divergence(component_model_type, 
			# 			structure['component_params'][c], structure['cluster_weights'][v], 
			# 			M_c, X_L, X_D, col=c, true_log_pdf=true_log_pdf, support=support)

			Q = [ (num_rows, c, x) for x in support ]

			# get predictive probabilities
			pred_probs = numpy.array(su.simple_predictive_probability(M_c, X_L, X_D, []*len(Q), Q))

			# KL is temporarily mean squared error
			KL[c,s] = numpy.mean((true_log_pdf-pred_probs)**2.0)

	kl_ret = numpy.mean(KL, axis=1)
	
	del mstate

	return kl_ret


if __name__ == '__main__':
    main()