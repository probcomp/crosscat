import crosscat.cython_code.State as State
import crosscat.utils.sample_utils as su
import crosscat.utils.data_utils as du

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext

import random
import pylab
import numpy

def test_one_feature_sampler(component_model_type):
    """
    Tests the ability of component model of component_model_type to capture the
    distribution of the data.
    1. Draws 100 random points from a standard normal distribution
    2. Initializes a component model with that data (and random hyperparameters)
    3. Draws data from that component model
    4. Initialize a crosscat state with that data
    5. Get one sample after 100 transitions
    6. Draw predictive samples
    7. Caluclates the 95 precent support of the continuous distribution or the 
        entire support of the discrete distribution
    8. Calculate the true pdf for each point in the support
    9. Calculate the predictive probability given the sample for each point in
        the support
    10. Plot the original data, predictive samples, pdf, and predictive 
        probabilities
    """
    N = 100
    
    get_next_seed = lambda : random.randrange(2147483647)
    
    X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])
    
    hyperparameters = component_model_type.draw_hyperparameters(X)[0]
    
    component_model = component_model_type.from_data(X, hyperparameters)
    
    model_parameters = component_model.sample_parameters_given_hyper()
    
    # generate data from the parameters
    T = component_model_type.generate_data_from_parameters(model_parameters, N, gen_seed=get_next_seed())

    # FIXME:
    # currently there is a bug that causes a freeze when a 1-feature crosscat
    # state is intialized so the below code is so we can test while we wait
    # for the bug fix
    T1 = component_model_type.generate_data_from_parameters(model_parameters, N, gen_seed=get_next_seed())
    T2 = component_model_type.generate_data_from_parameters(model_parameters, N, gen_seed=get_next_seed())
    T1 = numpy.array(T1)
    T2 = numpy.array(T2)
    T = numpy.hstack((T1, T2))
    # T = T.tolist()
    # END hack code
    
    # create a crosscat state 
    M_c = du.gen_M_c_from_T(T)
    state = State.p_State(M_c, T)
    
    # transitions
    state.transition(n_steps=100)
    
    # get the sample
    X_L = state.get_X_L()
    X_D = state.get_X_D()
    
    # generate samples
    predictive_samples = numpy.array(su.simple_predictive_sample(M_c, X_L, X_D, [], [(N,0)], get_next_seed, n=N))
    

    # get support
    discrete_support = component_model_type.generate_discrete_support(model_parameters, support=0.95, nbins=N)
    
    # calculate simple predictive probability for each point
    Q = [(N,0,x) for x in discrete_support]

    probabilities = su.simple_predictive_probability(M_c, X_L, X_D, []*len(Q), Q,)
    
    # plot normalized histogram of T and predictive samples
    pylab.hist([T[:,0],predictive_samples], bins=20, normed=True, label=["Original data", "Predictive samples"], color=["blue","red"], alpha=.7)
        
    # plot actual pdf of support given data params
    pylab.scatter(discrete_support, numpy.exp(component_model_type.log_pdf(numpy.array(discrete_support), model_parameters)), c="blue", label="true pdf", alpha=.7)
        
    # plot predictive probability of support points
    pylab.scatter(discrete_support, numpy.exp(probabilities), c="red", label="predictive probability", alpha=.7)
        
    pylab.legend()

    ylimits = pylab.gca().get_ylim()
    pylab.ylim([0,ylimits[1]])

    pylab.show()

if __name__ == '__main__':
    test_one_feature_sampler(ccmext.p_ContinuousComponentModel)