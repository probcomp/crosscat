import crosscat.cython_code.ContinuousComponentModel as ccm
import math
import random
import numpy
import six

from scipy.stats import norm as norm

from crosscat.utils.general_utils import logmeanexp

# FIXME: Using this instead of randrange because randrange is different before
# and after Python 3.2, and we hardcoded some likelihoods that depend on the
# RNG.
next_seed = lambda rng: int(rng.random() * 2147483647)

LOG_2 = math.log(2.0)
default_hyperparameters = dict(nu=1.0, mu=0.0, s=1.0, r=1.0)
default_data_parameters = dict(mu=0.0, rho=1.0)


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

    keys = ['mu', 'nu', 'r', 's']

    for key in keys:
        if key not in hypers.keys():
            raise KeyError("missing key in hypers: %r" % (key,))

    for key, value in six.iteritems(hypers):
        if key not in keys:
            raise KeyError("invalid hypers key: %r" % (key,))

        if not isinstance(value, (float, numpy.float64)):
            raise TypeError("%r should be float" % (key,))

        if key in ['nu', 'r', 's']:
            if value <= 0.0:
                raise ValueError("hypers[%r] should be greater than 0" % (key,))


def check_model_params_dict(params):
    if type(params) is not dict:
        raise TypeError("params should be a dict")

    keys = ['mu', 'rho']

    for key in keys:
        if key not in params:
            raise KeyError("missing key in params: %r" % (key,))

    for key, value in six.iteritems(params):
        if key not in keys:
            raise KeyError("invalid params key: %r" % (key,))

        if not isinstance(value, (float, numpy.float64)):
            raise TypeError("%r should be float" % (key,))

        if key == "rho":
            if value <= 0.0:
                raise ValueError("rho should be greater than 0")
        elif key != "mu":
            raise KeyError("Invalid params key: %r" % (key,))

###############################################################################
#   The class extension
###############################################################################
class p_ContinuousComponentModel(ccm.p_ContinuousComponentModel):

    model_type = 'normal_inverse_gamma'
    cctype = 'continuous'

    @classmethod
    def from_parameters(cls, N, data_params=default_data_parameters, hypers=None, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from random data.
        Inputs:
          N: the number of data points
          data_params: a dict with the following keys
              mu: the mean of the data
              rho: the precision of the data
          hypers: a dict with the following keys
              mu: the prior mean of the data
              s: hyperparameter
              r: hyperparameter
              nu: hyperparameter
          gen_seed: an integer from which the rng is seeded
        """

        check_model_params_dict(data_params)

        data_rho = data_params['rho']

        data_mean = data_params['mu']
        data_std = (1.0/data_rho)**.5

        rng = random.Random(gen_seed)
        X = [ [rng.normalvariate(data_mean, data_std)] for i in range(N)]
        X = numpy.array(X)
        check_data_type_column_data(X)

        if hypers is None:
            hypers = cls.draw_hyperparameters(X, n_draws=1, gen_seed=next_seed(rng))[0]

        check_hyperparams_dict(hypers)

        sum_x = numpy.sum(X)
        sum_x_squared = numpy.sum(X**2.0)

        hypers['fixed'] = 0.0

        return cls(hypers, float(N), sum_x, sum_x_squared)

    @classmethod
    def from_data(cls, X, hypers=None, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from data X
        Inputs:
            X: a column of data (numpy)
            hypers: dict with the following entries
                mu: the prior mean of the data
                s: hyperparameter
                r: hyperparameter
                nu: hyperparameter
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

        sum_x = numpy.sum(X)
        sum_x_squared = numpy.sum(X**2.0)

        hypers['fixed'] = 0.0

        return cls(hypers, float(N), sum_x, sum_x_squared)

    def sample_parameters_given_hyper(self, gen_seed=0):
        """
        Samples a Gaussian parameter given the current hyperparameters.
        Inputs:
            gen_seed: integer used to seed the rng
        """
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = random.Random(gen_seed)

        hypers = self.get_hypers()
        s = hypers[b's']
        r = hypers[b'r']
        nu = hypers[b'nu']
        m = hypers[b'mu']

        rho = rng.gammavariate(nu/2.0, s)
        mu = rng.normalvariate(m, (r/rho)**.5)

        assert(rho > 0)

        params = {'mu': mu, 'rho': rho}

        return params

    def uncollapsed_likelihood(self, X, parameters):
        """
        Calculates the score of the data X under this component model with mean
        mu and precision rho.
        Inputs:
            X: A column of data (numpy)
            parameters: a dict with the following keys
                mu: the Gaussian mean
                rho: the precision of the Gaussian
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        mu = parameters['mu']
        rho = parameters['rho']

        N = float(len(X))

        hypers = self.get_hypers()
        s = hypers[b's']
        r = hypers[b'r']
        nu = hypers[b'nu']
        m = hypers[b'mu']

        sum_err = numpy.sum((mu-X)**2.0)

        log_likelihood = self.log_likelihood(X, {'mu':mu, 'rho':rho})
        log_prior_mu = norm.logpdf(m, (r/rho)**.5)
        log_prior_rho = -(nu/2.0)*LOG_2+(nu/2.0)*math.log(s)+ \
            (nu/2.0-1.0)*math.log(rho)-.5*s*rho-math.lgamma(nu/2.0)

        log_p = log_likelihood + log_prior_mu + log_prior_rho

        return log_p

    @staticmethod
    def log_likelihood(X, parameters):
        """
        Calculates the log likelihood of the data X given mean mu and precision
        rho.
        Inputs:
            X: a column of data (numpy)
            parameters: a dict with the following keys
                mu: the Gaussian mean
                rho: the precision of the Gaussian
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        sigma = (1.0/parameters['rho'])**.5

        log_likelihood = numpy.sum(norm.logpdf(X,parameters['mu'],sigma))

        return log_likelihood

    @staticmethod
    def log_pdf(X, parameters):
        """
        Calculates the pdf for each point in the data X given mean mu and
        precision rho.
        Inputs:
            X: a column of data (numpy)
            parameters: a dict with the following keys
                mu: the Gaussian mean
                rho: the precision of the Gaussian
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        sigma = (1.0/parameters['rho'])**.5

        return norm.logpdf(X,parameters['mu'],sigma)

    @staticmethod
    def cdf(X, parameters):
        """
        Calculates the cdf for each point in the data X given mean mu and
        precision rho.
        Inputs:
            X: a column of data (numpy)
            parameters: a dict with the following keys
                mu: the Gaussian mean
                rho: the precision of the Gaussian
        """
        check_data_type_column_data(X)
        check_model_params_dict(parameters)

        sigma = (1.0/parameters['rho'])**.5

        return norm.cdf(X,parameters['mu'],sigma)

    def brute_force_marginal_likelihood(self, X, n_samples=10000, gen_seed=0):
        """
        Calculates the log marginal likelihood via brute force method in which
        parameters (mu and rho) are repeatedly drawn from the prior, the
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
            params: a dict with entries 'mu' and 'rho'
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
        sigma = (1.0/params['rho'])**.5

        interval = norm.interval(support,mu,sigma)

        a = interval[0]
        b = interval[1]

        support_range = b - a;
        support_bin_size = support_range/(nbins-1.0)

        bins = [a+i*support_bin_size for i in range(nbins)]

        return bins

    @staticmethod
    def draw_hyperparameters(X, n_draws=1, gen_seed=0):
        """
        Draws hyperparameters r, nu, mu, and s from the same distribution that
        generates the grid in the C++ code.
        Inputs:
             X: a column of data (numpy)
             n_draws: the number of draws
             gen_seed: seed the rng
        Output:
            A list of dicts of draws where each entry has keys 'mu', 'r', 'nu',
            and 's'.
        """
        check_data_type_column_data(X)
        if type(n_draws) is not int:
            raise TypeError("n_draws should be an int")
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")

        rng = random.Random(gen_seed)

        samples = []

        N = float(len(X))
        data_mean = numpy.sum(X)/N

        sum_sq_deviation = numpy.sum((data_mean-X)**2.0)

        nu_r_draw_range = (0.0, math.log(N))
        mu_draw_range = (numpy.min(X), numpy.max(X))
        s_draw_range = (sum_sq_deviation/100.0, sum_sq_deviation)

        for i in range(n_draws):
            nu = math.exp(rng.uniform(nu_r_draw_range[0], nu_r_draw_range[1]))
            r = math.exp(rng.uniform(nu_r_draw_range[0], nu_r_draw_range[1]))
            mu = rng.uniform(mu_draw_range[0], mu_draw_range[1])
            s = rng.uniform(s_draw_range[0], s_draw_range[1])

            this_draw = dict(nu=nu, r=r, mu=mu, s=s)

            samples.append(this_draw)

        assert len(samples) == n_draws

        return samples

    @staticmethod
    def generate_data_from_parameters(params, N, gen_seed=0):
        """
        Generates data from a gaussina distribution
        Inputs:
            params: a dict with entries 'mu' and 'rho'
            N: number of data points
        """
        if type(N) is not int:
            raise TypeError("N should be an int")

        if N <= 0:
            raise ValueError("N should be greater than 0")

        check_model_params_dict(params)

        mu = params['mu']
        sigma = (1.0/params['rho'])**.5

        rng = random.Random(gen_seed)
        X = numpy.array([[rng.normalvariate(mu, sigma)] for i in range(N)])

        assert len(X) == N

        return X

    @staticmethod
    def get_suffstat_names():
        """
        Returns a list of the names of the sufficient statistics
        """
        params = ['sum_x', 'sum_x_squared']
        return params

    @staticmethod
    def get_suffstat_bounds():
        """
        Returns a dict where each key-value pair is a sufficient statistic and a
        tuple with the lower and upper bounds
        """
        minf = float("-inf")
        inf = float("inf")
        params = dict(sum_x=(minf,inf), sum_x_squared=(0.0 ,inf))
        return params

    @staticmethod
    def get_hyperparameter_names():
        """
        Returns a list of the names of the prior hyperparameters
        """
        params = ['mu', 'nu', 'r', 's']
        return params

    @staticmethod
    def get_hyperparameter_bounds():
        """
        Returns a dict where each key-value pair is a hyperparameter and a
        tuple with the lower and upper bounds
        """
        minf = float("-inf")
        inf = float("inf")
        params = dict(mu=(minf,inf), nu=(0.0 ,inf), r=(0.0, inf), s=(0.0, inf))
        return params

    @staticmethod
    def get_model_parameter_names():
        """
        Returns a list of the names of the model parameters
        """
        params = ['mu', 'rho']
        return params

    @staticmethod
    def get_model_parameter_bounds():
        """
        Returns a dict where each key-value pair is a model parameter and a
        tuple with the lower and upper bounds
        """
        minf = float("-inf")
        inf = float("inf")
        params = dict(mu=(minf,inf), rho=(0.0 ,inf))
        return params
