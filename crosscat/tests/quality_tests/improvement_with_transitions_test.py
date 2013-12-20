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

import sys

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
	print "TEST IMPROVMENT OVER ITERATIONS (SINGLE COLUMN)"
	print " Tests if the error and KL divergence decrease and the ARI increases"
	print " as a function of the number of transisitions. Uses linear regression"
	print " to check the trend of the various statistics."
	print " "
	print " ** NOTE: This test will take a while to run for contiuous "
	print " distributions because it estimates KL divergence at each transition."
	unittest.main()

class TestCondifdence(unittest.TestCase):
	def setUp(self):
		self.show_plot=True

	def test_normal_inverse_gamma_predictive_sample_improves_over_iterations(self):
		improvement = test_improvement_over_transitions(
						ccmext.p_ContinuousComponentModel, 
						seed=0, show_plot=self.show_plot)
		assert improvement

	def test_dirchlet_multinomial_predictive_sample_improves_over_iterations(self):
		improvement = test_improvement_over_transitions(
			mcmext.p_MultinomialComponentModel, seed=0, show_plot=self.show_plot)
		assert improvement

def test_improvement_over_transitions(component_model_type, num_clusters=2,
	num_rows=100, separation=.9, num_samples=None, num_transitions=100, 
	seed=0, show_plot=True, print_out=True):
	""" Shows ARI and the error of impute over iterations.
	"""

	N = num_rows

	if num_samples is None:
		num_samples = multiprocessing.cpu_count()
	
	random.seed(seed)
	get_next_seed = lambda : random.randrange(2147483647)

	# generate a single column of data from the component_model 
	weights = [1.0/float(num_clusters)]*num_clusters
	cctype = component_model_type.cctype
	T, M_c, struc = sdg.gen_data([cctype], N, [0], [weights], [separation], 
				seed=get_next_seed(), distargs=[distargs[cctype]], 
				return_structure=True)

	T_array = numpy.array(T)

	X = numpy.zeros((N,num_transitions))
	KL = numpy.zeros((num_samples, num_transitions))
	ARI = []


	support = qtu.get_mixture_support(cctype, component_model_type, 
					struc['component_params'][0], nbins=500, support=.9999)
	true_log_pdf = qtu.get_mixture_pdf(support, component_model_type, 
					struc['component_params'][0],weights)

	M_r = du.gen_M_r_from_T(T)
	mstate = mpe.MultiprocessingEngine()
	X_L_list, X_D_list = mstate.initialize(M_c, M_r, T, n_chains=num_samples)

	teststr = "improvement over num_transitions (%s)" % cctype

	itr = 0
	for i in range(num_transitions):
		# transition
		X_L_list, X_D_list = mstate.analyze(M_c, T, X_L_list, X_D_list, n_steps=1)

		for s in range(num_samples):
			itr += 1
			qtu.print_progress(itr,num_transitions*num_samples, teststr)
			# get partitions and generate a predictive column
			X_L = X_L_list[s]
			X_D = X_D_list[s]

			T_inf = sdg.predictive_columns(M_c, X_L, X_D, [0], 
					seed=get_next_seed())

			if cctype == 'multinomial':
				K = distargs[cctype]['K']
				weights = numpy.zeros(numpy.array(K))
				for params in struc['component_params'][0]:
					weights += numpy.array(params['weights'])*(1.0/num_clusters)
				weights *= float(N)
				inf_hist = qtu.bincount(T_inf, bins=range(K))
				err, _ = stats.power_divergence(inf_hist, weights, lambda_='pearson')
				err = numpy.ones(N)*err
			else:
				err = (T_array-T_inf)**2.0

			KL[s,i] = qtu.KL_divergence(component_model_type, 
						struc['component_params'][0], weights, M_c, X_L, X_D,
						true_log_pdf=true_log_pdf, support=support)

			for j in range(N):
				X[j,i] += err[j]

		ARI.append(ctu.multi_chain_ARI(X_L_list, X_D_list, [0],
			struc['rows_to_clusters'])[0])

	X /= num_samples

	# mean and standard error
	X_mean = numpy.mean(X,axis=0)
	X_err = numpy.std(X,axis=0)/float(num_samples)**.5

	KL_mean = numpy.mean(KL, axis=0)
	KL_err = numpy.std(KL, axis=0)/float(num_samples)**.5

	if show_plot:
		pylab.subplot(1,3,1)
		pylab.errorbar(range(num_transitions), X_mean, yerr=X_err)
		pylab.xlabel('iteration')
		pylab.ylabel('error across each data point')
		pylab.title('error of predictive sample over iterations, N=%i' % N)

		pylab.subplot(1,3,2)
		pylab.errorbar(range(num_transitions), KL_mean, yerr=KL_err)
		pylab.xlabel('iteration')
		pylab.ylabel('KL divergence')
		pylab.title('KL divergence, N=%i' % N)

		pylab.subplot(1,3,3)
		pylab.errorbar(range(num_transitions), ARI, yerr=KL_err)
		pylab.xlabel('iteration')
		pylab.ylabel('ARI table mean')
		pylab.title('ARI table mean, N=%i' % N)

		pylab.show()

	# fit regression lines to the change over time
	lr_err = stats.linregress(range(num_transitions),X_mean)
	lr_KL = stats.linregress(range(num_transitions),KL_mean)
	lr_ARI = stats.linregress(range(num_transitions),ARI)

	# KL divergence should decrease over time
	error_decreased = lr_err[0] < 0 and lr_err[3] < .05
	KL_divergence_decreased = lr_KL[0] < 0 and lr_KL[3] < .05
	ARI_increased = lr_ARI[0] > 0 and lr_ARI[3] < .05

	if print_out:
		print " "
		print "======================================"
		print "CHANGE OVER TIME (SINGLE COLUMN)"
		print "TEST INFORMATION:"
		print "       data type: " + component_model_type.cctype
		print "        num_rows: " + str(num_rows)
		print "    num_clusters: " + str(num_clusters)
		print "     num_samples: " + str(num_samples)
		print " num_transitions: " + str(num_transitions)
		print "RESULTS (IMPUTE ERROR REGRESSION):"
		print "     slope: " + str(lr_err[0])
		print "   r-value: " + str(lr_err[2])
		print "   p-value: " + str(lr_err[3])
		print "    PASSED: " + str(error_decreased)
		print "RESULTS (KL DIVERGENCE REGRESSION):"
		print "     slope: " + str(lr_KL[0])
		print "   r-value: " + str(lr_KL[2])
		print "   p-value: " + str(lr_KL[3])
		print "    PASSED: " + str(KL_divergence_decreased)
		print "RESULTS (ARI REGRESSION):"
		print "     slope: " + str(lr_ARI[0])
		print "   r-value: " + str(lr_ARI[2])
		print "   p-value: " + str(lr_ARI[3])
		print "    PASSED: " + str(ARI_increased)

	del mstate
	# ARI doesn't make a lot of sense of a single column problem with 
	# categorical data
	if is_categorical[cctype]:
		return error_decreased and KL_divergence_decreased
	else:
		# the KL divergence estimator sometimes returns negative numbers
		# so I am not including it in the test. 
		return error_decreased and ARI_increased

if __name__ == '__main__':
    main()