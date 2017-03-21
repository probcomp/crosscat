import crosscat.cython_code.State as State
import crosscat.utils.sample_utils as su
import crosscat.utils.data_utils as du

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext
import crosscat.tests.component_model_extensions.CyclicComponentModel as cycmext
import crosscat.tests.synthetic_data_generator as sdg
import crosscat.tests.quality_test_utils as qtu

import matplotlib
matplotlib.use('Agg')

import random
import pylab
import numpy
import math

import unittest
import pdb
import six

from scipy import stats

distargs = dict(
    multinomial=dict(K=5),
    continuous=None,
    cyclic=None,
    )

default_data_parameters = dict(
    symmetric_dirichlet_discrete=dict(weights=[1.0/5.0]*5),
    normal_inverse_gamma=dict(mu=0.0, rho=1.0),
    vonmises=dict(mu=math.pi, kappa=2.0)
    )

is_discrete = dict(
    symmetric_dirichlet_discrete=True,
    normal_inverse_gamma=False,
    vonmises=False
    )


def main():
    unittest.main()

class TestComponentModelQuality(unittest.TestCase):
    def setUp(self):
        self.show_plot = True

    def test_normal_inverse_gamma_model(self):
        assert(check_one_feature_mixture(ccmext.p_ContinuousComponentModel,
                show_plot=self.show_plot) > .1)

    def test_dirchlet_multinomial_model(self):
        assert(check_one_feature_mixture(mcmext.p_MultinomialComponentModel,
                show_plot=self.show_plot) > .1)

    # Github issue #50
    # https://github.com/probcomp/crosscat/issues/50
    @unittest.expectedFailure
    def test_vonmises_vonmises_model(self):
        assert(check_one_feature_mixture(cycmext.p_CyclicComponentModel,
                show_plot=self.show_plot) > .1)


def get_params_string(params):
    string = dict()
    for k,v in six.iteritems(params):
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

def check_one_feature_mixture(component_model_type, num_clusters=3, show_plot=False, seed=None):
    """

    """
    random.seed(seed)

    N = 300
    separation = .9
    
    get_next_seed = lambda : random.randrange(2147483647)

    cluster_weights = [[1.0/float(num_clusters)]*num_clusters]

    cctype = component_model_type.cctype
    T, M_c, structure = sdg.gen_data([cctype], N, [0], cluster_weights,
                        [separation], seed=get_next_seed(),
                        distargs=[distargs[cctype]],
                        return_structure=True)


    T_list = list(T)
    T = numpy.array(T)
    
    # pdb.set_trace()    
    # create a crosscat state 
    M_c = du.gen_M_c_from_T(T_list, cctypes=[cctype])
    
    state = State.p_State(M_c, T_list)

    # Get support over all component models
    discrete_support = qtu.get_mixture_support(cctype, component_model_type,
                         structure['component_params'][0], nbins=250)
    
    # calculate simple predictive probability for each point
    Q = [(N,0,x) for x in discrete_support]

    # transitions
    state.transition(n_steps=200)

    # get the sample
    X_L = state.get_X_L()
    X_D = state.get_X_D()
    
    # generate samples
    # kstest has doesn't compute the same answer with row and column vectors
    # so we flatten this column vector into a row vector.
    predictive_samples = sdg.predictive_columns(M_c, X_L, X_D, [0],
                            seed=get_next_seed()).flatten(1)
    

    probabilities = su.simple_predictive_probability(M_c, X_L, X_D, []*len(Q), Q)
    
    # get histogram. Different behavior for discrete and continuous types. For some reason
    # the normed property isn't normalizing the multinomial histogram to 1.
    # T = T[:,0]
    if is_discrete[component_model_type.model_type]:
        bins = list(range(len(discrete_support)))
        T_hist = numpy.array(qtu.bincount(T, bins=bins))
        S_hist = numpy.array(qtu.bincount(predictive_samples, bins=bins))
        T_hist = T_hist/float(numpy.sum(T_hist))
        S_hist = S_hist/float(numpy.sum(S_hist))
        edges = numpy.array(discrete_support,dtype=float)
    else:
        T_hist, edges = numpy.histogram(T, bins=min(50,len(discrete_support)), normed=True)
        S_hist, _ =  numpy.histogram(predictive_samples, bins=edges, normed=True)
        edges = edges[0:-1]

    # Goodness-of-fit-tests
    if not is_discrete[component_model_type.model_type]:
        # do a KS tests if the distribution in continuous
        # cdf = lambda x: component_model_type.cdf(x, model_parameters)
        # stat, p = stats.kstest(predictive_samples, cdf)   # 1-sample test
        stat, p = stats.ks_2samp(predictive_samples, T[:,0]) # 2-sample test
        test_str = "KS"
    else:
        # Cressie-Read power divergence statistic and goodness of fit test.
        # This function gives a lot of flexibility in the method <lambda_> used.
        freq_obs = S_hist*N
        freq_exp = numpy.exp(probabilities)*N
        stat, p = stats.power_divergence(freq_obs, freq_exp, lambda_='pearson')
        test_str = "Chi-square"
    
    if show_plot:
        pylab.clf()
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

        filename = component_model_type.model_type + "_mixtrue.png"
        pylab.savefig(filename)
        pylab.close()

    return p

if __name__ == '__main__':
    main()
