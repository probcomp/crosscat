import crosscat.tests.quality_test_utils as qtu

import pylab
import numpy

from scipy.stats import norm 
from scipy.stats import pearsonr 

import unittest

class TestKLDivergence(unittest.TestCase):
	def test_kl_divergence_estimate_correlates_higly_to_analytical(self):
		assert test_KL_divergence_for_normal_distributions(show_plot=True) < .000001


def main():
	unittest.main()

def actual_KL(m1,s1,m2,s2):
	return numpy.log(s2/s1) + (s1**2.0+(m1-m2)**2.0)/(2*s2**2.0) - .5

def test_KL_divergence_for_normal_distributions(show_plot=True):

	mu_0 = 0
	sigma_0 = 1

	interval = norm.interval(.99,mu_0,sigma_0)

	support = numpy.linspace(interval[0], interval[1], num=2000)

	mus = numpy.linspace(0, 3, num=30)

	p_0 = norm.logpdf(support, mu_0, sigma_0)

	KL_inf = []
	KL_ana = []

	for mu in mus:
		p_1 = norm.logpdf(support, mu, sigma_0)

		kld = qtu.KL_divergence_arrays(support, p_0, p_1, False)

		KL_inf.append(float(kld))
		KL_ana.append(actual_KL(mu_0, sigma_0, mu, sigma_0))


	KL_inf = numpy.array(KL_inf)
	KL_ana = numpy.array(KL_ana)
	KL_diff = KL_ana-KL_inf


	if show_plot:
		pylab.subplot(1,2,1)
		pylab.plot(KL_inf, label='est')
		pylab.plot(KL_ana, label='analytical')
		pylab.title('estimated KL')
		pylab.legend()

		pylab.subplot(1,2,2)
		pylab.plot(KL_diff)
		pylab.title('KL error')

		pylab.show()


	_, p = pearsonr(KL_inf, KL_ana)

	return p


if __name__ == '__main__':
    main()