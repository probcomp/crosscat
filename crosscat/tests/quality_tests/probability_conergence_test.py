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

def main():
	print " "
	print "======================================================================="
	print "TEST PROBABILITY IMPROVEMNT OVER TRANSITIONS"
	print " Checks that the probability of new data points becomes closer"
	print " to the true probability over iterations."
	print " "
	unittest.main()

class TestP(unittest.TestCase):
	def setUp(self):
		self.show_plot=True

	def test_normal_inverse_gamma_probability_converges(self):
		improvement = test_probability_convergence(
						ccmext.p_ContinuousComponentModel, 
						seed=0, show_plot=self.show_plot)
		assert improvement

	def test_dirichlet_multinomial_probability_converges(self):
		improvement = test_probability_convergence(
						mcmext.p_MultinomialComponentModel, 
						seed=0, show_plot=self.show_plot)
		assert improvement

def test_probability_convergence(component_model_type, num_clusters=2,
	num_rows=1000, separation=.9, num_samples=None, num_transitions=200, 
	seed=0, show_plot=True, print_out=True):
	""" 
	FIXE: ADD DOC
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

	n_data_points = 10

	# get some hypothetical data from the support
	support = qtu.get_mixture_support(cctype, component_model_type, 
					struc['component_params'][0], nbins=500, support=.95)

	if len(support) <= n_data_points:
		X_h = support
	else:
		X_h = numpy.array(random.sample(support, n_data_points))

	true_log_pdf = numpy.exp(qtu.get_mixture_pdf(X_h, component_model_type, 
					struc['component_params'][0], weights))

	# init the sampler
	M_r = du.gen_M_r_from_T(T)
	mstate = mpe.MultiprocessingEngine(cpu_count=multiprocessing.cpu_count())
	X_L_list, X_D_list = mstate.initialize(M_c, M_r, T, n_chains=num_samples)


	abs_error_P = numpy.zeros((num_transitions, len(true_log_pdf)))
	teststr = "pdf improvement over num_transitions (%s)" % cctype
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

			# generate simple predictive probability queries
			Q = []
			for x in X_h:
				Q.append((num_rows,0,x))

			inferred_log_pdf = numpy.array(su.simple_predictive_probability(M_c, X_L, X_D, [], Q)).flatten(1)

			abs_error_P[i,:] += numpy.abs(true_log_pdf-numpy.exp(inferred_log_pdf))

	abs_error_P /= float(num_samples)

	if show_plot:
		for i in range(len(true_log_pdf)):
			pylab.plot(range(num_transitions), abs_error_P[:,i], color="black",
					alpha=.3, linewidth=1)
		pylab.plot(range(num_transitions), numpy.mean(abs_error_P, axis=1),
					color="red", alpha=1, linewidth=4, label="mean")
		pylab.xlabel('iteration')
		pylab.ylabel('ABS pdf_true-pdf_crosscat')
		pylab.title('ABS pdf_true-pdf_crosscat, N=%i' % N)
		pylab.legend()

		pylab.show()

	# fit regression lines to the change over time
	lr_err = stats.linregress(range(num_transitions), numpy.mean(abs_error_P, axis=1))
	

	if print_out:
		print "======================================"
		print "ABS(TRUE_PDF-CC_PDF) (SINGLE COLUMN)"
		print "TEST INFORMATION:"
		print "       data type: " + component_model_type.cctype
		print "        num_rows: " + str(num_rows)
		print "    num_clusters: " + str(num_clusters)
		print "     num_samples: " + str(num_samples)
		print " num_transitions: " + str(num_transitions)
		print "RESULTS (TRUE LOGP - CC LOGP ERROR REGRESSION):"
		print "     slope: " + str(lr_err[0])
		print "   r-value: " + str(lr_err[2])
		print "   p-value: " + str(lr_err[3])

	# KL divergence should decrease over time
	error_decreased = lr_err[0] < 0 and lr_err[3] < .1

	return error_decreased

if __name__ == '__main__':
    main()