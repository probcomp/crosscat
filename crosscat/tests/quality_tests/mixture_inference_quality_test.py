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
import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext
import crosscat.tests.quality_tests.synthetic_data_generator as sdg
import crosscat.tests.quality_tests.quality_test_utils as qtu

import random
import pylab
import numpy

import unittest

import crosscat.MultiprocessingEngine as mpe
import multiprocessing

from scipy import stats

distargs = dict(
    multinomial=dict(K=5),
    continuous=None,
    )

default_data_parameters = dict(
    symmetric_dirichlet_discrete=dict(weights=[1.0/5.0]*5),
    normal_inverse_gamma=dict(mu=0.0, rho=1.0)
    )

is_discrete = dict(
    symmetric_dirichlet_discrete=True,
    normal_inverse_gamma=False
    )


def main():
    print " "
    print "======================================================================="
    print "TEST MIXTURE INFERENCE QUALITY"
    print " Performs a 2-sample KS or Chi-square test for a single column"
    print " problem with multiple clusters."
    print " "
    print " ** NOTE: Used primarily for testing new data types."
    unittest.main()

class TestComponentModelQuality(unittest.TestCase):
    def setUp(self):
        self.show_plot = False

    def test_normal_inverse_gamma_model(self):
        assert(test_one_feature_mixture(ccmext.p_ContinuousComponentModel, 
                show_plot=self.show_plot, print_out=True) > .1)

    def test_dirchlet_multinomial_model(self):
        assert(test_one_feature_mixture(mcmext.p_MultinomialComponentModel, 
                show_plot=self.show_plot, print_out=True) > .1)


def get_params_string(params):
    string = dict()
    for k,v in params.iteritems():
        if isinstance(v, float):
            string[k] = round(v,3)
        elif isinstance(v, list):
            string[k] = [round(val,3) for val in v]

    return str(string)

def cdf_array(X, component_model):
    cdf = numpy.zeros(len(X))
    for i in range(len(X)):
        x = X[i]
        cdf[i] = component_model.get_predictive_cdf(x,[])

    assert i == len(X)-1
    assert i > 0
    return cdf

def test_one_feature_mixture(component_model_type, num_clusters=2, num_rows=500,
        separation=.9, seed=None, show_plot=False, print_out=False):
    """
        test_one_feature_mixture
        tests the inferred distribution (crosscat) to the original distribution
        using KS (continuous) and Chi-square (discrete) tests. Returns p
    """
    random.seed(seed)

    N = num_rows
    
    get_next_seed = lambda : random.randrange(2147483647)

    # uniform cluster weights
    cluster_weights = [[1.0/float(num_clusters)]*num_clusters]

    cctype = component_model_type.cctype
    T, M_c, structure = sdg.gen_data([cctype], N, [0], cluster_weights,
                        [separation], seed=get_next_seed(),
                        distargs=[distargs[cctype]],
                        return_structure=True)

    T = numpy.array(T)
    T_list = T
    
    # one sample for each processor
    num_samples = multiprocessing.cpu_count()

    # create a crosscat state 
    M_c = du.gen_M_c_from_T(T_list, cctypes=[cctype])
    M_r = du.gen_M_r_from_T(T_list)
    
    mstate = mpe.MultiprocessingEngine(cpu_count=num_samples)
    X_L_list, X_D_list = mstate.initialize(M_c, M_r, T_list, n_chains=num_samples)
    
    # transitions
    n_transitions=400
    X_L_list, X_D_list = mstate.analyze(M_c, T_list, X_L_list, X_D_list,
                            n_steps=n_transitions)
    
    all_stats = []
    all_ps = []

    teststr = "single column mixture inference test (%s)" % cctype

    for chain in range(num_samples):

        qtu.print_progress(chain, num_samples, teststr)

        X_L = X_L_list[chain]
        X_D = X_D_list[chain]
        # generate samples
        # kstest has doesn't compute the same answer with row and column vectors
        # so we flatten this column vector into a row vector.
        predictive_samples = sdg.predictive_columns(M_c, X_L, X_D, [0],
                                seed=get_next_seed()).flatten(1)
        
        # Get support over all component models
        discrete_support = qtu.get_mixture_support(cctype, component_model_type,
                             structure['component_params'][0], nbins=500)

        # calculate simple predictive probability for each point
        Q = [(N,0,x) for x in discrete_support]

        probabilities = su.simple_predictive_probability(M_c, X_L, X_D, []*len(Q), Q)
        
        # get histogram. Different behavior for discrete and continuous types. For some reason
        # the normed property isn't normalizing the multinomial histogram to 1.
        if is_discrete[component_model_type.model_type]:
            bins = range(len(discrete_support))
            T_hist = numpy.array(qtu.bincount(T, bins=bins))
            S_hist = numpy.array(qtu.bincount(predictive_samples, bins=bins))
            T_hist = T_hist/float(numpy.sum(T_hist))
            S_hist = S_hist/float(numpy.sum(S_hist))
            edges = numpy.array(discrete_support,dtype=float)
        else:
            T_hist, edges = numpy.histogram(T, bins=min(20,len(discrete_support)), normed=True)
            S_hist, _ =  numpy.histogram(predictive_samples, bins=edges, normed=True)
            edges = edges[0:-1]

        # Goodness-of-fit-tests
        if not is_discrete[component_model_type.model_type]:
            # do a KS tests if the distribution in continuous
            # cdf = lambda x: component_model_type.cdf(x, model_parameters)
            # stat, p = stats.kstest(predictive_samples, cdf)   # 1-sample test
            stat, p = stats.ks_2samp(predictive_samples, T[:,0]) # 2-sample test
            test_str = "2-sample KS"
        else:
            # Cressie-Read power divergence statistic and goodness of fit test.
            # This function gives a lot of flexibility in the method <lambda_> used.
            freq_obs = S_hist*N
            freq_exp = numpy.exp(probabilities)*N
            stat, p = stats.power_divergence(freq_obs, freq_exp, lambda_='pearson')
            test_str = "Chi-square"


        all_stats.append(stat)
        all_ps.append(p)

    if show_plot:
        lpdf = qtu.get_mixture_pdf(discrete_support, component_model_type, 
                structure['component_params'][0], [1.0/num_clusters]*num_clusters)
        pylab.axes([0.1, 0.1, .8, .7])
        # bin widths
        width = (numpy.max(edges)-numpy.min(edges))/len(edges)
        pylab.bar(edges, T_hist, color='blue', alpha=.5, width=width, label='Original data', zorder=1)
        pylab.bar(edges, S_hist, color='red', alpha=.5, width=width, label='Predictive samples', zorder=2)

        # plot actual pdf of support given data params
        pylab.scatter(discrete_support, 
            numpy.exp(lpdf), 
            c="blue", 
            edgecolor="none",
            s=100, 
            label="true pdf", 
            alpha=1,
            zorder=3)
                
        # plot predictive probability of support points
        pylab.scatter(discrete_support, 
            numpy.exp(probabilities), 
            c="red", 
            edgecolor="none",
            s=100, 
            label="predictive probability", 
            alpha=1,
            zorder=4)
            
        pylab.legend()

        ylimits = pylab.gca().get_ylim()
        pylab.ylim([0,ylimits[1]])

        title_string = "%i samples drawn from %i %s components: \ninference after 200 crosscat transitions\n%s test: p = %f" \
            % (N, num_clusters, component_model_type.cctype, test_str, round(p,4))

        pylab.title(title_string, fontsize=12)

        pylab.show()

    if print_out:
        print " "
        print "======================================"
        print "MIXTURE INFERENCE (SINGLE COLUMN)"
        print "TEST INFORMATION:"
        print "       data type: " + component_model_type.cctype
        print "        num_rows: " + str(num_rows)
        print "    num_clusters: " + str(num_clusters)
        print "     num_samples: " + str(num_samples)
        print " num_transitions: " + str(n_transitions)
        print "      separation: " + str(separation)
        print "RESULTS (%s) for each chain" % test_str 
        print "   statistic(s): " + str([ "%.4f" % stat for stat in all_stats])
        print "           p(s): " + str([ "%.4f" % p for p in all_ps])

    del mstate
    return numpy.mean(all_ps)

if __name__ == '__main__':
    main()
