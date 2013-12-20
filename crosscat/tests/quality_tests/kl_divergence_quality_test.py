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

# This test just makes sure that my KL divergence estimator is correct
# test the estimator vs the analylytical form for two normal dsitributions
import crosscat.tests.quality_tests.quality_test_utils as qtu

import pylab
import numpy

from scipy.stats import norm 
from scipy.stats import pearsonr 

import unittest

class TestKLDivergence(unittest.TestCase):
	def test_kl_divergence_estimate_correlates_higly_to_analytical(self):
		assert test_KL_divergence_for_normal_distributions(show_plot=False, ) < .000001


def main():
	unittest.main()

def actual_KL(m1,s1,m2,s2):
	return numpy.log(s2/s1) + (s1**2.0+(m1-m2)**2.0)/(2*s2**2.0) - .5

def test_KL_divergence_for_normal_distributions(show_plot=False, print_out=True):

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

	r, p = pearsonr(KL_inf, KL_ana)

	if print_out:
		print "================================================="
		print "KL DIVERGENCE ESTIMATOR (TWO GAUSSIANS)"
		print "TEST INFORMATION:"
		print "  KL divergence estimator vs analytical formula"
		print "  for two Gaussian distributions."
		print "  The test is repeated several for means Gaussians"
		print "  with increaseing distance between their means."
		print "  Correlation is run on the resulting KL "
		print "  divergences from the formula and the estimator."
		print "RESULTS (PEARSON R):"
		print "  r: " + str(r)
		print "  p: " + str(p)

	return p


if __name__ == '__main__':
    main()