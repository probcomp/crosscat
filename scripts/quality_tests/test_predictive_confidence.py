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
	multinomial=dict(K=8),
	continuous=None,
	)

def main():
    unittest.main()

class TestCondifdence(unittest.TestCase):
	def setUp(self):
		self.show_plot=True

	def test_normal_inverse_gamma_predictive_sample_improves_over_iterations(self):
		improvement = check_predictive_sample_improvement(
						ccmext.p_ContinuousComponentModel, 
						seed=0, show_plot=self.show_plot)
		assert improvement

	def test_dirchlet_multinomial_predictive_sample_improves_over_iterations(self):
		improvement = check_predictive_sample_improvement(
			mcmext.p_MultinomialComponentModel, seed=0, show_plot=self.show_plot)
		assert improvement

def check_predictive_sample_improvement(component_model_type, seed=0, show_plot=True):
	""" Shows the error of predictive sample over iterations.
	"""

	num_transitions = 100
	num_samples = 10	
	num_clusters = 2
	separation = .9	# cluster separation
	N = 150
	
	random.seed(seed)
	get_next_seed = lambda : random.randrange(2147483647)

	# generate a single column of data from the component_model 
	cctype = component_model_type.cctype
	T, M_c, struc = sdg.gen_data([cctype], N, [0], [[.5,.5]], [separation], 
				seed=get_next_seed(), distargs=[distargs[cctype]], 
				return_structure=True)

	T_array = numpy.array(T)

	X = numpy.zeros((N,num_transitions))
	KL = numpy.zeros((num_samples, num_transitions))


	support = qtu.get_mixture_support(cctype, component_model_type, 
					struc['component_params'][0], nbins=1000, support=.995)
	true_log_pdf = qtu.get_mixture_pdf(support, component_model_type, 
					struc['component_params'][0],[.5,.5])

	for s in range(num_samples):
		# generate the state
		state = State.p_State(M_c, T, SEED=get_next_seed())

		for i in range(num_transitions):
			# transition
			state.transition()

			# get partitions and generate a predictive column
			X_L = state.get_X_L()
			X_D = state.get_X_D()

			T_inf = sdg.predictive_columns(M_c, X_L, X_D, [0], 
					seed=get_next_seed())

			if cctype == 'multinomial':
				K = distargs[cctype]['K']
				weights = numpy.zeros(numpy.array(K))
				for params in struc['component_params'][0]:
					weights += numpy.array(params['weights'])*(1.0/num_clusters)
				weights *= float(N)
				inf_hist = qtu.bincount(T_inf, bins=list(range(K)))
				err, _ = stats.power_divergence(inf_hist, weights, lambda_='pearson')
				err = numpy.ones(N)*err
			else:
				err = (T_array-T_inf)**2.0

			KL[s,i] = qtu.KL_divergence(component_model_type, 
						struc['component_params'][0], [.5,.5], M_c, X_L, X_D,
						true_log_pdf=true_log_pdf, support=support)

			for j in range(N):
				X[j,i] += err[j]

	X /= num_samples

	# mean and standard error
	X_mean = numpy.mean(X,axis=0)
	X_err = numpy.std(X,axis=0)/float(num_samples)**.5

	KL_mean = numpy.mean(KL, axis=0)
	KL_err = numpy.std(KL, axis=0)/float(num_samples)**.5

	if show_plot:
		pylab.subplot(1,2,1)
		pylab.errorbar(list(range(num_transitions)), X_mean, yerr=X_err)
		pylab.xlabel('iteration')
		pylab.ylabel('error across each data point')
		pylab.title('error of predictive sample over iterations, N=%i' % N)

		pylab.subplot(1,2,2)
		pylab.errorbar(list(range(num_transitions)), KL_mean, yerr=KL_err)
		pylab.xlabel('iteration')
		pylab.ylabel('KL divergence')
		pylab.title('KL divergence, N=%i' % N)

		pylab.show()

	# error should decrease over time
	return X_mean[0] > X_mean[-1] and KL_mean[0] > KL_mean[-1]

if __name__ == '__main__':
    main()
