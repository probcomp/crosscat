import crosscat.utils.sample_utils as su

import numpy
import math

import pdb

from scipy.misc import logsumexp

is_discrete = {
    'multinomial' : True,
    'ordinal' : True,
    'continuous' : False
    }

def get_mixture_pdf(X, component_model_class, parameters_list, component_weights):
    """ FIXME: Add doc
    """

    if not isinstance(X, numpy.ndarray) and not isinstance(X, list):
        raise TypeError("X should be a list or numpy array of data")

    if not isinstance(parameters_list, list):
        raise TypeError('parameters_list should be a list')

    if not isinstance(component_weights, list):
        raise TypeError('component_weights should be a lsit')

    if len(parameters_list) != len(component_weights):
        raise ValueError("parameters_list and component_weights should have the\
            same number of elements")

    if math.fabs(sum(component_weights)-1.0) > .0000001:
        raise ValueError("component_weights should sum to 1")

    for w in component_weights:
        assert w >= 0.0

    K = len(component_weights)

    lpdf = numpy.zeros((K,len(X)))

    for k in range(K):
        if component_weights[k] == 0.0:
            lp = 0
        else:
            lp = math.log(component_weights[k])+component_model_class.log_pdf(X,
                    parameters_list[k])

        lpdf[k,:] = lp

    lpdf = logsumexp(lpdf,axis=0)

    assert len(lpdf) == len(X)

    return lpdf

def bincount(X, bins=None):
    """ Counts the elements in X according to bins.
        Inputs:
            - X: A 1-D list or numpt array or integers.
            - bins: (optional): a list of elements. If bins is None, bins will
                be range range(min(X), max(X)+1). If bins is provided, bins
                must contain at least each element in X
        Outputs: 
            - counts: a list of the number of element in each bin
        Ex:
            >>> import quality_test_utils as qtu
            >>> X = [0, 1, 2, 3]
            >>> qtu.bincount(X)
            [1, 1, 1, 1]
            >>> X = [1, 2, 2, 4, 6]
            >>> qtu.bincount(X)
            [1, 2, 0, 1, 0, 1]
            >>> bins = range(7)
            >>> qtu.bincount(X,bins)
            [0, 1, 2, 0, 1, 0, 1]
            >>> bins = [1,2,4,6]
            >>> qtu.bincount(X,bins)
            [1, 2, 1, 1]
    """

    if not isinstance(X, list) and not isinstance(X, numpy.ndarray):
        raise TypeError('X should be a list or a numpy array')

    if isinstance(X, numpy.ndarray):
        if len(X.shape) > 1:
            if X.shape[1] != 1:
                raise ValueError('X should be a vector')

    Y = numpy.array(X, dtype=int)
    if bins is None:
        minval = numpy.min(Y)
        maxval = numpy.max(Y)

        bins = list(range(minval, maxval+1))

    if not isinstance(bins, list):
        raise TypeError('bins should be a list')

    counts = [0]*len(bins)

    for y in Y:
        bin_index = bins.index(y)
        counts[bin_index] += 1

    assert len(counts) == len(bins)
    assert sum(counts) == len(Y)

    return counts

def get_mixture_support(cctype, component_model_class, parameters_list, support=.95, nbins=100):
    """
    """
    if cctype == 'multinomial':
        discrete_support = component_model_class.generate_discrete_support(
                            parameters_list[0])
    elif cctype == 'cyclic':
        discrete_support = numpy.linspace(0,2*math.pi,nbins)
    else:
        for k in range(len(parameters_list)):
            model_parameters = parameters_list[k]
            support_k = numpy.array(component_model_class.generate_discrete_support(
                        model_parameters, support=support))
            if k == 0:
                all_support = support_k
            else:
                all_support = numpy.hstack((all_support, support_k))

        discrete_support = numpy.linspace(numpy.min(all_support), 
                            numpy.max(all_support), num=nbins)

        assert len(discrete_support) == nbins

    return numpy.array(discrete_support)

def KL_divergence(component_model_class, parameters_list, component_weights,
    M_c, X_L, X_D, n_samples=1000, true_log_pdf=None, support=None):
    """ FIXME: Add doc
    """

    # FIXME: Add validation code

    cctype = component_model_class.cctype

    # get support (X)
    if support is None:
        support = get_mixture_support(cctype, component_model_class, parameters_list, 
                nbins=n_samples, support=.995)
    elif not isinstance(support, numpy.ndarray):
        raise TypeError("support must be a numpy array (vector)")

    # get true pdf
    if true_log_pdf is None:
        true_log_pdf = get_mixture_pdf(support, component_model_class, parameters_list,
                    component_weights)
    elif not isinstance(true_log_pdf, numpy.ndarray):
        raise TypeError("true_log_pdf should be a numpy array (vector)")

    row = len(X_D[0])
    Q = [ (row,0,x) for x in support ]

    # get predictive probabilities
    pred_probs = su.simple_predictive_probability(M_c, X_L, X_D, []*len(Q), Q)

    kld = KL_divergence_arrays(support, pred_probs, true_log_pdf,
            is_discrete[cctype])

    return float(kld)

def KL_divergence_arrays(support, log_true, log_inferred, is_discrete):
    """
        Separated this function from KL_divergence for testing purposes.
        Inputs:
            - support: numpy array of support intervals
            - log_true: log pdf at support for the "true" distribution
            - log_inferred: log pdf at support for the distribution to test against the
              "true" distribution
            - is_discrete: is this a discrete variable True/False
        Returns:
            - KL divergence
    """

    # KL divergence formula, recall X and Y are log
    F = (log_true-log_inferred)*numpy.exp(log_true)
    if is_discrete:
        kld = numpy.sum(F)
    else:
        # trapezoidal quadrature
        intervals = numpy.diff(support)
        fs = F[:-1] + (numpy.diff(F) / 2.0)
        kld = numpy.sum(intervals*fs)

    return kld

