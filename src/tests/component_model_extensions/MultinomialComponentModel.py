import crosscat.cython_code.MultinomialComponentModel as mcm
import math
import random
import numpy
import six

from scipy.special import gammaln as gammaln

from crosscat.utils.general_utils import logmeanexp

next_seed = lambda rng: int(rng.uniform() * 2147483647)

###############################################################################
#   Input-checking and exception-handling functions
###############################################################################
def check_type_force_float(x, name):
    """
    If an int is passed, convert it to a float. If some other type is passed,
    raise an exception.
    """
    if type(x) is int:
        return float(x)
    elif type(x) is not float and type(x) is not numpy.float64:
        raise TypeError("%r should be a float" % (name,))
    else:
        return x

def counts_to_data(counts, rng):
    """
    Converts a vector of counts to data.
    """
    assert type(counts) is list or type(counts) is numpy.ndarray
    K = len(counts)
    N = int(sum(counts))
    X = []
    for k in range(K):
        i = 0
        while i < counts[k]:
            X.append([k])
            i += 1

        assert i == counts[k]

    assert len(X) == N

    random.shuffle(X, rng.uniform)
    X = numpy.array(X, dtype=float)

    return X


def check_data_type_column_data(X):
    """
    Makes sure that X is a numpy array and that it is a column vector
    """
    if type(X) is list:
        X = numpy.array(X)

    if type(X) is not numpy.ndarray:
        raise TypeError("X should be type numpy.ndarray or a list")

    if len(X.shape) == 2 and X.shape[1] > 1:
        raise TypeError("X should have a single column.")


def check_model_parameters_dict(model_parameters_dict):

    if type(model_parameters_dict) is not dict:
        raise TypeError("model_parameters_dict should be a dict")

    keys = ['weights']

    for key in keys:
        if key not in model_parameters_dict:
            raise KeyError("model_parameters_dict should have key %r" % (key,))

    for key, value in six.iteritems(model_parameters_dict):
        if key == "weights":
            if type(value) is not list:
                raise TypeError("model parameters dict key 'weights' should be a list")
            if type(value[0]) is list:
                raise TypeError("weights should not be a list of lists, should be a list of floats")
            if math.fabs(sum(value) - 1.0) > .00000001:
                raise ValueError("model parameters dict key 'weights' should sum to 1.0")
        else:
            raise KeyError("invalid key, %r, for model parameters dict" % (key,))

def check_hyperparameters_dict(hyperparameters_dict):

    # 'fixed' key is not necessary for user-defined hyperparameters
    keys = ['dirichlet_alpha', 'K']

    for key in keys:
        if key not in hyperparameters_dict:
            raise KeyError("hyperparameters_dict should have key %r" % (key,))

    for key, value in six.iteritems(hyperparameters_dict):
        if key == "K":
            if type(value) is not int:
                raise TypeError("hyperparameters dict entry K should be an int")

            if value < 1:
                raise ValueError("hyperparameters dict entry K should be greater than 0")
        elif key == "dirichlet_alpha":
            if not isinstance(value, (float, numpy.float64, int)):
                raise TypeError("hyperparameters dict entry dirichlet_alpha should be a float or int")
            if value <= 0.0:
                raise ValueError("hyperparameters dict entry dirichlet_alpha should be greater than 0")
        elif key == "fixed":
            pass
        else:
            raise KeyError("invalid key, %r, for hyperparameters dict" % (key,))

def check_data_vs_k(X,K):
    if type(X) is numpy.ndarray:
        X = X.flatten(1)
        X = X.tolist()
    K_data = len(set(X))
    if K_data > K:
        raise ValueError("the number of items in the data is greater than K")

###############################################################################
#   The class extension
###############################################################################
class p_MultinomialComponentModel(mcm.p_MultinomialComponentModel):

    model_type = 'symmetric_dirichlet_discrete'
    cctype = 'multinomial'

    @classmethod
    def from_parameters(cls, N, params=None, hypers=None, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from random data.
        Inputs:
          N: the number of data points
          params: a dict with the following keys
              weights: a K-length list that sums to 1.0
          hypers: a dict with the following keys
              K: the number of categories
              dirichlet_alpha: Dirichlet alpha parameter. The distribution is
              symmetric so only one value is needed
          gen_seed: an integer from which the rng is seeded
        """

        if type(N) is not int:
            raise TypeError("N should be an int")
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        # if the parameters dict or the hypers dict exist, validate them
        if params is not None:
            check_model_parameters_dict(params)

        if hypers is not None:
            check_hyperparameters_dict(hypers)

        rng = numpy.random.RandomState(gen_seed)

        # get the number of categories
        if params is None:
            if hypers is None:
                K = int(N/2.0)
            else:
                K = int(hypers['K'])
            weights = rng.uniform(size=(1,K))
            weights = weights/numpy.sum(weights)
            weights = weights.tolist()[0]
            assert len(weights) == K
            params = dict(weights=weights)

            check_model_parameters_dict(params)
        else:
            K = len(params['weights'])
            if hypers:
                if K != hypers['K']:
                    raise ValueError("K in params does not match K in hypers")

        # generate synthetic data
        counts = numpy.array(rng.multinomial(N, params['weights']), dtype=int)

        X = counts_to_data(counts, rng)

        check_data_type_column_data(X)

        # generate the sufficient statistics
        suffstats = dict()
        for k in range(K):
            suffstats[str(k)] = counts[k]

        if hypers is None:
            hypers = cls.draw_hyperparameters(X, n_draws=1, gen_seed=next_seed(rng))[0]
            check_hyperparameters_dict(hypers)

        # hypers['K'] = check_type_force_float(hypers['K'], "hypers['K']")
        hypers['dirichlet_alpha'] = check_type_force_float(hypers['dirichlet_alpha'], "hypers['dirichlet_alpha']")

        # add fixed parameter to hyperparameters
        hypers['fixed'] = 0.0

        suffstats = {'counts':suffstats}

        return cls(hypers, float(N), **suffstats)

    @classmethod
    def from_data(cls, X, hypers=None, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from data X
        Inputs:
            X: a column of data (numpy)
            hypers: a dict with the following keys
              K: the number of categories
              dirichlet_alpha: Dirichlet alpha parameter. The distribution is
              symmetric so only one value is needed
            gen_seed: a int to seed the rng
        """
        # FIXME: Figure out a way to accept a list of strings
        check_data_type_column_data(X)
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = numpy.random.RandomState(gen_seed)

        if hypers is None:
            hypers = cls.draw_hyperparameters(X, gen_seed=next_seed(rng))[0]
            check_hyperparameters_dict(hypers)
        else:
            check_hyperparameters_dict(hypers)
            K = hypers['K']
            check_data_vs_k(X,K)

        hypers['dirichlet_alpha'] = check_type_force_float(hypers['dirichlet_alpha'], "hypers['dirichlet_alpha']")

        N = len(X)
        K = hypers['K']

        counts = [0]*K
        for x in X:
            try:
                counts[int(x)] += 1
            except IndexError:
                raise IndexError

        # generate the sufficient statistics
        suffstats = dict()
        for k in range(int(K)):
            suffstats[str(k)] = counts[k]

        suffstats = {'counts':suffstats}

        hypers['fixed'] = 0.0

        return cls(hypers, float(N), **suffstats)

    def sample_parameters_given_hyper(self, gen_seed=0):
        """
        Samples weights given the current hyperparameters
        Inputs:
            gen_seed: integer used to seed the rng
        """
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = numpy.random.RandomState(gen_seed)

        hypers = self.get_hypers()
        dirichlet_alpha = hypers[b'dirichlet_alpha']
        K = hypers[b'K']

        alpha = numpy.array([dirichlet_alpha]*int(K))

        weights = rng.dirichlet(alpha)
        weights = weights.tolist()

        params = {'weights': weights}

        return params

    def uncollapsed_likelihood(self, X, params):
        """
        Calculates the score of the data X under this component model with mean
        mu and precision rho.
        Inputs:
            X: A column of data (numpy)
            params: a dict with the following keys
                weights: a list of category weights (should sum to 1)
        """
        check_data_type_column_data(X)
        check_model_parameters_dict(params)

        hypers = self.get_hypers()

        assert len(params['weights']) == int(hypers[b'K'])

        dirichlet_alpha = hypers[b'dirichlet_alpha']
        K = float(hypers[b'K'])
        check_data_vs_k(X,K)

        weights = numpy.array(params['weights'])

        log_likelihood = self.log_likelihood(X, params)
        logB = gammaln(dirichlet_alpha)*K - gammaln(dirichlet_alpha*K)
        log_prior = -logB + numpy.sum((dirichlet_alpha-1.0)*numpy.log(weights))

        log_p = log_likelihood + log_prior

        return log_p

    @staticmethod
    def log_likelihood(X, params):
        """
        Calculates the log likelihood of the data X given mean mu and precision
        rho.
        Inputs:
            X: a column of data (numpy)
            params: a dict with the following keys
                weights: a list of categories weights (should sum to 1)
        """
        check_data_type_column_data(X)
        check_model_parameters_dict(params)

        N = len(X)
        K = len(params['weights'])
        check_data_vs_k(X,K)
        counts= numpy.bincount(X,minlength=K)

        weights = numpy.array(params['weights'])

        A = gammaln(N+1)-numpy.sum(gammaln(counts+1))
        B = numpy.sum(counts*numpy.log(weights));

        log_likelihood = A+B

        return log_likelihood

    @staticmethod
    def log_pdf(X, params):
        """
        Calculates the log pdf of the data X given mean mu and precision
        rho.
        Inputs:
            X: a column of data (numpy)
            params: a dict with the following keys
                weights: a list of categories weights (should sum to 1)
        """
        check_data_type_column_data(X)
        check_model_parameters_dict(params)

        N = len(X)

        weights = numpy.array(params['weights'])

        lpdf = []
        for x in X:
            w = weights[int(x)]
            if w == 0.0 or w == 0:
                lpdf.append(float('-Inf'))
            else:
                lpdf.append(math.log(w))

        return numpy.array(lpdf)

    def brute_force_marginal_likelihood(self, X, n_samples=10000, gen_seed=0):
        """
        Calculates the log marginal likelihood via brute force method in which
        parameters (weights) are repeatedly drawn from the prior, the
        likelihood is calculated for each set of parameters, then the average is
        taken.
        Inputs:
            X: A column of data (numpy)
            n_samples: the number of draws
            gen_Seed: seed for the rng
        """
        check_data_type_column_data(X)

        if type(n_samples) is not int:
            raise TypeError("n_samples should be an int")
        if n_samples <= 0:
            raise ValueError("n_samples should be greater than 0")
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        hypers = self.get_hypers()
        K = hypers[b'K']
        check_data_vs_k(X,K)

        rng = numpy.random.RandomState(gen_seed)
        log_likelihoods = [0]*n_samples
        for i in range(n_samples):
            params = self.sample_parameters_given_hyper(gen_seed=next_seed(rng))
            log_likelihoods[i] = self.log_likelihood(X, params)

        log_marginal_likelihood = logmeanexp(log_likelihoods)

        return log_marginal_likelihood

    @staticmethod
    def generate_discrete_support(params):
        """
        Returns the a sequential list of the number of categories
        Inputs:
            params: a dict with entries 'mu' and 'rho'
        """
        check_model_parameters_dict(params)

        return list(range(len(params['weights'])))


    @staticmethod
    def draw_hyperparameters(X, n_draws=1, gen_seed=0):
        """
        Draws hyperparameters dirichlet_alpha from the same distribution that
        generates the grid in the C++ code.
        Inputs:
             X: a column of data or an int which acts as K. If a data array is
                provided, K is assumed to be max(X)+1
             n_draws: the number of draws
             gen_seed: seed the rng
        Output:
            A list of dicts of draws where each entry has keys 'dirichlet_alpha'
            and 'K'. K is defined by the data and will be the same for each samples
        """
        if type(X) is list or type(X) is numpy.ndarray:
            check_data_type_column_data(X)
            K = int(max(X)+1)
        elif type(X) is int:
            if X < 1:
                raise ValueError("If X is an int, it should be greatert than 1")
            K = X
        else:
            raise TypeError("X should be an array or int")

        if type(n_draws) is not int:
            raise TypeError("n_draws should be an int")

        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = numpy.random.RandomState(gen_seed)

        samples = []

        # get draw ranges
        alpha_draw_range = (0.1, math.log(K))


        for i in range(n_draws):
            alpha = math.exp(rng.uniform(alpha_draw_range[0], alpha_draw_range[1]))

            this_draw = dict(dirichlet_alpha=alpha, K=K)

            samples.append(this_draw)

        assert len(samples) == n_draws

        return samples

    @staticmethod
    def generate_data_from_parameters(params, N, gen_seed=0):
        """
        returns a set of intervals over which the component model pdf is
        supported.
        Inputs:
            params: a dict with entries 'weights'
            N: number of data points
        """
        if type(N) is not int:
            raise TypeError("N should be an int")

        if N <= 0:
            raise ValueError("N should be greater than 0")

        if type(params) is not dict:
            raise TypeError("params should be a dict")

        rng = numpy.random.RandomState(gen_seed)

        check_model_parameters_dict(params)

        # multinomial draw
        counts = numpy.array(rng.multinomial(N, params['weights']), dtype=int)

        X = counts_to_data(counts, rng)

        assert len(X) == N

        return X
