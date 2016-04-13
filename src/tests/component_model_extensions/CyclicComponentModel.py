import crosscat.cython_code.CyclicComponentModel as ccm
import math
import random
import numpy
import six

from scipy.stats import vonmises

from crosscat.utils.general_utils import logmeanexp

import pdb

pi = math.pi

next_seed = lambda rng: rng.randrange(2147483647)

default_hyperparameters = dict(a=1.0, b=pi, kappa=4.0)
default_data_parameters = dict(mu=pi, kappa=4.0)

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
    elif not isinstance(x, (float, numpy.float64)):
        raise TypeError("%r should be a float" % (name,))
    else:
        return x

def check_data_type_column_data(X):
    """
    Makes sure that X is a numpy array and that it is a column vector
    """
    if type(X) is not numpy.ndarray:
        raise TypeError("X should be type numpy.ndarray")

    if len(X.shape) == 2 and X.shape[1] > 1:
        raise TypeError("X should have a single column.")

def check_hyperparams_dict(hypers):
    if type(hypers) is not dict:
        raise TypeError("hypers should be a dict")

    keys = ['a', 'b', 'kappa']

    for key in keys:
        if key not in hypers:
            raise KeyError("missing key in hypers: %r" % (key,))

    for key, value in six.iteritems(hypers):
        if key not in keys:
            raise KeyError("invalid hypers key: %r" % (key,))

        if not isinstance(value, (float, numpy.float64)):
            raise TypeError("%r should be float" % (key,))

        if key in ['a', 'kappa']:
            if value <= 0.0:
                raise ValueError("hypers[%r] should be greater than 0" % (key,))

        if key == 'b':
            if value <= 0.0 or value >= 2*pi:
                raise ValueError("hypers[%r] should be in [0,2*pi]" % (key,))



def check_model_params_dict(params):
    if type(params) is not dict:
        raise TypeError("params should be a dict")

    keys = ['mu', 'kappa']

    for key in keys:
        if key not in params:
            raise KeyError("missing key in params: %r" % (key,))

    for key, value in six.iteritems(params):
        if key not in keys:
            raise KeyError("invalid params key: %r" % (key,))

        if not isinstance(value, (float, numpy.float64)):
            raise TypeError("%r should be float" % (key,))

        if key == "kappa":
            if value <= 0.0:
                raise ValueError("kappa should be greater than 0")
        elif key != "mu":
            raise KeyError("Invalid params key: %r" % (key,))
        else:
            if value < 0.0 or value > 2*pi:
                raise ValueError("mu should be in [0,2*pi]")

###############################################################################
#   The class extension
###############################################################################
class p_CyclicComponentModel(ccm.p_CyclicComponentModel):

    model_type = 'vonmises'
    cctype = 'cyclic'

    @classmethod
    def from_parameters(cls, N, data_params=default_data_parameters, hypers=None, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from random data.
        Inputs:
          N: the number of data points
          data_params: a dict with the following keys
              mu: the mean of the data
              kappa: the precision of the data
          hypers: a dict with the following keys
              a: the prior precision of the mean
              b: the prior mean of the
              kappa: precision parameter
          gen_seed: an integer from which the rng is seeded
        """

        check_model_params_dict(data_params)

        data_kappa = data_params['kappa']
        data_mean = data_params['mu']

        rng = random.Random(gen_seed)
        X = [ [rng.vonmisesvariate(data_mean-math.pi, data_kappa)+math.pi] for i in range(N)]
        X = numpy.array(X)
        check_data_type_column_data(X)

        if hypers is None:
            hypers = cls.draw_hyperparameters(X, n_draws=1, gen_seed=next_seed(rng))[0]

        check_hyperparams_dict(hypers)

        sum_sin_x = numpy.sum(numpy.sin(X))
        sum_cos_x = numpy.sum(numpy.cos(X))

        hypers['fixed'] = 0.0

        return cls(hypers, float(N), sum_sin_x, sum_cos_x)

    @classmethod
    def from_data(cls, X, hypers=None, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from data X
        Inputs:
            X: a column of data (numpy)
            hypers: dict with the following entries
                a: the prior precision of the mean
                b: the prior mean of the
                kappa: precision parameter
            gen_seed: a int to seed the rng
        """
        check_data_type_column_data(X)
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = random.Random(gen_seed)

        if hypers is None:
            hypers = cls.draw_hyperparameters(X, gen_seed=next_seed(rng))[0]

        check_hyperparams_dict(hypers)

        N = len(X)

        sum_sin_x = numpy.sum(numpy.sin(X))
        sum_cos_x = numpy.sum(numpy.cos(X))

        hypers['fixed'] = 0.0

        return cls(hypers, float(N), sum_sin_x, sum_cos_x)

    def sample_parameters_given_hyper(self, gen_seed=0):
        """
        Samples a Gaussian parameter given the current hyperparameters.
        Inputs:
            gen_seed: integer used to seed the rng
        """
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        nprng = numpy.random.RandomState(gen_seed)

        hypers = self.get_hypers()
        a = hypers['a']
        b = hypers['b']
        kappa = hypers['kappa']

        mu = nprng.vonmises(b-math.pi, a)+math.pi
        kappa = hypers['kappa']

        assert(kappa > 0)
        assert(mu >= 0 and mu <= 2*pi)

        params = {'mu': mu, 'kappa': kappa}

        return params

    def uncollapsed_likelihood(self, X, parameters):
        """
        Calculates the score of the data X under this component model with mean
        mu and precision kappa.
        Inputs:
            X: A column of data (numpy)
            parameters: a dict with the following keys
                mu: the Von Mises mean
                kappa: the precision of the Von Mises
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        mu = parameters['mu']
        kappa = parameters['kappa']

        N = float(len(X))

        hypers = self.get_hypers()
        a = hypers['a']
        b = hypers['b']
        kappa = hypers['kappa']

        sum_err = numpy.sum((mu-X)**2.0)

        log_likelihood = self.log_likelihood(X, {'mu':mu, 'kappa':rho})
        log_prior_mu = vonmises.logpdf(b, a)

        log_p = log_likelihood + log_prior_mu + log_prior_rho

        return log_p

    @staticmethod
    def log_likelihood(X, parameters):
        """
        Calculates the log likelihood of the data X given mean mu and precision
        kappa.
        Inputs:
            X: a column of data (numpy)
            parameters: a dict with the following keys
                mu: the Von Mises mean
                kappa: the precision of the Von Mises
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        log_likelihood = numpy.sum(vonmises.logpdf(X-math.pi, parameters['mu']-math.pi, parameters['kappa']))

        return log_likelihood

    @staticmethod
    def log_pdf(X, parameters):
        """
        Calculates the pdf for each point in the data X given mean mu and
        precision kappa.
        Inputs:
            X: a column of data (numpy)
            parameters: a dict with the following keys
                mu: the Von Mises mean
                kappa: the precision of the Von Mises
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        return vonmises.logpdf(X--math.pi, parameters['kappa'],loc=parameters['mu']-math.pi)

    @staticmethod
    def cdf(X, parameters):
        """
        Calculates the cdf for each point in the data X given mean mu and
        precision kappa.
        Inputs:
            X: a column of data (numpy)
            parameters: a dict with the following keys
                mu: the Von Mises mean
                kappa: the precision of the Von Mises
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        return vonmises.cdf(X-math.pi, parameters['mu']-math.pi, parameters['kappa'])

    def brute_force_marginal_likelihood(self, X, n_samples=10000, gen_seed=0):
        """
        Calculates the log marginal likelihood via brute force method in which
        parameters (mu and kappa) are repeatedly drawn from the prior, the
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

        N = float(len(X))
        rng = random.Random(gen_seed)
        log_likelihoods = [0]*n_samples
        for i in range(n_samples):
            params = self.sample_parameters_given_hyper(gen_seed=next_seed(rng))
            log_likelihoods[i] = self.log_likelihood(X, params)

        log_marginal_likelihood = logmeanexp(log_likelihoods)

        return log_marginal_likelihood

    @staticmethod
    def generate_discrete_support(params, support=0.95, nbins=100):
        """
        returns a set of intervals over which the component model pdf is
        supported.
        Inputs:
            params: a dict with entries 'mu' and 'kappa'
            nbins: cardinality of the set or the number of grid points in the
                approximation
            support: a float in (0,1) that describes the amount of probability
                we want in the range of support
        """
        if type(nbins) is not int:
            raise TypeError("nbins should be an int")

        if nbins <= 0:
            raise ValueError("nbins should be greater than 0")

        support = check_type_force_float(support, "support")
        if support <= 0.0 or support >= 1.0:
            raise ValueError("support is a float st: 0 < support < 1")

        check_model_params_dict(params)

        mu = params['mu']
        kappa = params['kappa']

        assert(mu >= 0 and mu <= 2*math.pi)

        a, b = vonmises.interval(support, kappa)
        a += mu
        b += mu
        assert -math.pi <= a < b <= 3*math.pi
        assert b - a <= 2*math.pi

        support_range = b - a;
        support_bin_size = support_range/(nbins-1.0)

        bins = [a+i*support_bin_size for i in range(nbins)]

        return bins

    @staticmethod
    def draw_hyperparameters(X, n_draws=1, gen_seed=0):
        """
        Draws hyperparameters a, b, and kappa from the same distribution that
        generates the grid in the C++ code.
        Inputs:
             X: a column of data (numpy)
             n_draws: the number of draws
             gen_seed: seed the rng
        Output:
            A list of dicts of draws where each entry has keys 'a', 'b', 'kappa'.
        """
        check_data_type_column_data(X)
        if type(n_draws) is not int:
            raise TypeError("n_draws should be an int")
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = random.Random(gen_seed)

        samples = []

        N = float(len(X))

        vx = numpy.var(X)

        a_kappa_draw_range = (vx, vx/N)
        mu_draw_range = (0, 2*pi)

        for i in range(n_draws):
            a = math.exp(rng.uniform(a_kappa_draw_range[0], a_kappa_draw_range[1]))
            kappa = math.exp(rng.uniform(a_kappa_draw_range[0], a_kappa_draw_range[1]))
            b = rng.uniform(mu_draw_range[0], mu_draw_range[1])

            this_draw = dict(a=a, b=b, kappa=kappa)

            samples.append(this_draw)

        assert len(samples) == n_draws

        return samples

    @staticmethod
    def generate_data_from_parameters(params, N, gen_seed=0):
        """
        Generates data from a gaussina distribution
        Inputs:
            params: a dict with entries 'mu' and 'kappa'
            N: number of data points
        """
        if type(N) is not int:
            raise TypeError("N should be an int")

        if N <= 0:
            raise ValueError("N should be greater than 0")

        nprng = numpy.random.RandomState(gen_seed)

        check_model_params_dict(params)

        mu = params['mu']
        kappa = params['kappa']

        X = numpy.array([[nprng.vonmises(mu-math.pi, kappa)+math.pi] for i in range(N)])

        for x in X:
            if x < 0. or x > 2.*math.pi:
                pdb.set_trace()
        assert len(X) == N

        return X

    @staticmethod
    def get_model_parameter_bounds():
        """
        Returns a dict where each key-value pair is a model parameter and a
        tuple with the lower and upper bounds
        """
        inf = float("inf")
        params = dict(mu=(0.0,2*pi), rho=(0.0 ,inf))
        return params
