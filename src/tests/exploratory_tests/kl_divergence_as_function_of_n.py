
import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.synthetic_data_generator as sdg
import crosscat.tests.quality_test_utils as qtu
import crosscat.utils.data_utils as du
import crosscat.MultiprocessingEngine as mpe

import random
import numpy
import pylab

import pdb

def test_kl_divergence_as_a_function_of_N_and_transitions():

	n_clusters = 3
	n_chains = 8
	do_times = 4

	# N_list = [25, 50, 100, 250, 500, 1000, 2000]
	N_list = [25, 50, 100, 175, 250, 400, 500]

	# max_transitions = 500
	max_transitions = 500
	transition_interval = 50
	t_iterations = max_transitions/transition_interval

	cctype = 'continuous'
	cluster_weights = [1.0/float(n_clusters)]*n_clusters
	separation = .5

	get_next_seed = lambda : random.randrange(2147483647)

	# data grid
	KLD = numpy.zeros((len(N_list), t_iterations+1))

	for _ in range(do_times):
		for n in range(len(N_list)):
			N = N_list[n]
			T, M_c, struc = sdg.gen_data([cctype], N, [0], [cluster_weights], 
							[separation], seed=get_next_seed(), distargs=[None],
							return_structure=True)

			M_r = du.gen_M_r_from_T(T)

			# precompute the support and pdf to speed up calculation of KL divergence
			support = qtu.get_mixture_support(cctype, 
						ccmext.p_ContinuousComponentModel, 
						struc['component_params'][0], nbins=1000, support=.995)
			true_log_pdf = qtu.get_mixture_pdf(support,
						ccmext.p_ContinuousComponentModel, 
						struc['component_params'][0],cluster_weights)

			# intialize a multiprocessing engine
			mstate = mpe.MultiprocessingEngine(cpu_count=8)
			X_L_list, X_D_list = mstate.initialize(get_next_seed(), M_c, M_r, T,
				n_chains=n_chains)

			# kl_divergences
			klds = numpy.zeros(len(X_L_list))

			for i in range(len(X_L_list)):
				X_L = X_L_list[i]
				X_D = X_D_list[i]
				KLD[n,0] += qtu.KL_divergence(ccmext.p_ContinuousComponentModel,
						struc['component_params'][0], cluster_weights, M_c, 
						X_L, X_D, n_samples=1000, support=support, 
						true_log_pdf=true_log_pdf)


			# run transition_interval then take a reading. Rinse and repeat.
			for t in range( t_iterations ):
				X_L_list, X_D_list = mstate.analyze(get_next_seed(), M_c, T,
							X_L_list, X_D_list, n_steps=transition_interval)

				for i in range(len(X_L_list)):
					X_L = X_L_list[i]
					X_D = X_D_list[i]
					KLD[n,t+1] += qtu.KL_divergence(ccmext.p_ContinuousComponentModel,
							struc['component_params'][0], cluster_weights, M_c, 
							X_L, X_D, n_samples=1000, support=support, 
							true_log_pdf=true_log_pdf)


	KLD /= float(n_chains*do_times)

	pylab.subplot(1,3,1)
	pylab.contourf(list(range(0,max_transitions+1,transition_interval)), N_list, KLD)
	pylab.title('KL divergence')
	pylab.ylabel('N')
	pylab.xlabel('# transitions')


	pylab.subplot(1,3,2)
	m_N = numpy.mean(KLD,axis=1)
	e_N = numpy.std(KLD,axis=1)/float(KLD.shape[1])**-.5
	pylab.errorbar(N_list,  m_N, yerr=e_N)
	pylab.title('KL divergence by N')
	pylab.xlabel('N')
	pylab.ylabel('KL divergence')

	pylab.subplot(1,3,3)
	m_t = numpy.mean(KLD,axis=0)
	e_t = numpy.std(KLD,axis=0)/float(KLD.shape[0])**-.5
	pylab.errorbar(list(range(0,max_transitions+1,transition_interval)), m_t, yerr=e_t)
	pylab.title('KL divergence by transitions')
	pylab.xlabel('trasition')
	pylab.ylabel('KL divergence')

	pylab.show()

	return KLD

if __name__ == '__main__':
	test_kl_divergence_as_a_function_of_N_and_transitions()
