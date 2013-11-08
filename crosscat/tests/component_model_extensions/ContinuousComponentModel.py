import crosscat.cython_code.ContinuousComponentModel as ccm
import math
import random
import sys
import pdb

from scipy.misc import logsumexp as logsumexp
from scipy.stats import norm as norm

LOG_2 = math.log(2.0)

def check_type_force_float(x, name):
    """
    If an int is passed, convert it to a float. If some other type is passed, 
    raise an exception.
    """
    if type(x) is int:
        return float(x)
    elif type(x) is not float:
        raise TypeError("%s should be a float" % name)
    else:
        return x


class p_ContinuousComponentModel(ccm.p_ContinuousComponentModel):
    
    type = "Normal Normal-Gamma"
    
    @classmethod
    def from_parameters(cls, N, data_mean=0.0, data_std=1.0, m=0.0, s=1.0, r=1.0, nu=1.0, gen_seed=0):
        """
        Initialize a continuous component model with sufficient statistics
        generated from random data.
        Inputs:
          N: the number of data points
          data_mean: the mean of the data
          data_std: the standard deviation of the data
          m: the prior mean of the data
          s: hyperparameter
          r: hyperparameter
          nu: hyperparameter
          gen_seed: an integer from which the rng is seeded
        """
        
        if type(N) is not int:
            raise TypeError("N should be an int")
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")
        data_mean = check_type_force_float(data_mean, "data_mean")
        data_std = check_type_force_float(data_std, "data_std")
        if data_std <= 0.0:
            raise ValueError("data_std should be greater than 0")
        m = check_type_force_float(m, "m")
        s = check_type_force_float(s, "s")
        if s <= 0.0:
            raise ValueError("s should be greater than 0")
        r = check_type_force_float(r, "r")
        nu = check_type_force_float(nu, "nu")
        if nu <= 0.0:
            raise ValueError("nu should be greater than 0")
            
        random.seed(gen_seed)
        X = [random.normalvariate(data_mean, data_std) for i in range(N)]
        sum_x = 0
        sum_x_squared = 0
        for n in range(N):
            sum_x += X[n]
            sum_x_squared += X[n]*X[n]
        
        hypers = {'mu': m, 's':s, 'r':r, 'nu':nu, 'fixed': 0.0}           
                        
        return cls(hypers, float(N), sum_x, sum_x_squared)
        
    def sample_parameters_given_hyper(self, gen_seed=0):
        """
        Samples a Gaussian parameter given the current hyperparameters.
        Inputs:
            which_parameter: either 'mu' (mean) or 'rho' (precision)
            gen_seed: integer used to seed the rng
        Outputs:
            The value of the sampled parameter
        """
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")
            
        random.seed(gen_seed)
        
        hypers = self.get_hypers()
        s = hypers['s']
        r = hypers['r']
        nu = hypers['nu']
        m = hypers['mu']
        
        rho = random.gammavariate(nu/2.0, s)
        mu = random.normalvariate(m, (r/rho)**.5)
        
        assert(rho > 0)
        
        params = {'mu': mu, 'rho': rho}
        
        return params
        
    def uncollapsed_likelihood(self, X, mu, rho):
        """
        Calculates the score of the data X under this component model with mean 
        mu and precision rho. 
        Inputs:
            X: A list of data
            mu: the component model mean
            rho: the component model precision
        """
        if type(X) is not list:
            raise TypeError("X should be a list")
        mu = check_type_force_float(mu, "mu")
        rho = check_type_force_float(rho, "rho")
        if rho <= 0.0:
            raise ValueError("rho should be greater than 0")
    
        N = float(len(X))
        
        hypers = self.get_hypers()
        s = hypers['s']
        r = hypers['r']
        nu = hypers['nu']
        m = hypers['mu']
        
        sum_err = 0
        for i in range(N):
            sum_err += (mu-X[i])**2.0
            
        log_likelihood = log_likelihood(self, X, mu, rho)   
        log_prior_mu = norm.logpdf(m, (r/rho)**.5)
        log_prior_rho = -(nu/2.0)*LOG_2+(nu/2.0)*math.log(s)+ \
            (nu/2.0-1.0)*math.log(rho)-.5*s*rho-math.lgamma(nu/2.0)
            
        log_p = log_likelihood + log_prior_mu + log_prior_rho
        
        return log_p
                
    @staticmethod
    def log_likelihood(X, mu, rho):
        """
        Calculates the log likelihood of the data X given mean mu and precision
        rho.
        Inputs:
            X: a list of data
            mu: the Gaussian mean
            rho: the precision of the Gaussian
        """
        if type(X) is not list:
            raise TypeError("X should be a list")
                        
        mu = check_type_force_float(mu, "mu")
        rho = check_type_force_float(rho, "rho")
        
        if rho <= 0.0:
            raise ValueError("rho should be greater than 0")
        
        sigma = (1.0/rho)**.5
        
        return sum(norm.logpdf(X,mu,sigma))
        
    def brute_force_marginal_likelihood(self, X, n_samples=10000, gen_seed=0):
        """
        Calculates the log marginal likelihood via brute force method in which
        parameters (mu and rho) are repeatedly drawn from the prior, the 
        likelihood is calculated for each set of parameters, then the average is
        taken.
        Inputs:
            X: A list of data
            n_samples: the number of draws
            gen_Seed: seed for the rng
        """
        if type(X) is not list:
            raise TypeError("X should be a list")
        if type(n_samples) is not int:
            raise TypeError("n_samples should be an int")
        if n_samples <= 0:
            raise ValueError("n_samples should be greater than 0")
        if type(gen_seed) is not int:
            raise TypeError("gen_seed should be an int")
            
        N = float(len(X))
        random.seed(gen_seed)
        log_likelihoods = [0]*n_samples        
        next_seed = lambda : random.randrange(2147483647)
        for i in range(n_samples):
            params = self.sample_parameter_given_hyper(next_seed())
            mu = params['mu']
            rho = params['rho']
            log_likelihoods[i] = self.log_likelihood(X, mu, rho)
            
        log_marginal_likelihood = logsumexp(log_likelihoods) - math.log(N)
        
        return log_marginal_likelihood
        
    @staticmethod
    def generate_discrete_support(params, support=0.95, nbins=100):
        """
        returns a set of intervals over which the component model pdf is 
        supported. 
        Inputs:
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
            
        if type(params) is not dict:
            raise TypeError("params should be a dict")
            
        for key, value in params.iteritems():
            if key == 'mu':
                params[key] = check_type_force_float(params[key], "mu")
            elif key == 'rho':
                params[key] = check_type_force_float(params[key], "rho")
                if value <= 0.0:
                    raise ValueError("params['rho'] should be greater than 0")
            else:
                raise ValueError("params should contain only the keys 'mu' and 'rho'")
        
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
    def draw_hyperparameters(X, n_draws, gen_seed=0):
        """
        Draws hyperparameters r, nu, mu, and s from the same distribution that 
        generates the grid in the C++ code.
        Inputs:
             X: a list of data
             n_draws: the number of draws
             gen_seed: seed the rng
        Output:
            A list of dicts of draws where each entry has keys 'mu', 'r', 'nu', 
            and 's'
        """
        if type(X) is not list:
            raise TypeError("X should be a list")
        if type(n_draws) is not int:
            raise TypeError("n_draws should be an int")
        if type(gen_seed) is not list:
            raise TypeError("gen_seed should be an int")
        
        random.seed(gen_seed)
        
        samples = []
        
        N = float(len(X))
        data_mean = sum(X)/N
        
        sum_sq_deviation = 0
        for i in range(N):
            sum_sq_deviation += (data_mean-X[i])**2.0
            
        nu_r_draw_range = (0.0, math.log(N))
        mu_draw_range = (min(X), max(X))
        s_draw_range = (sum_sq_deviation/100.0, sum_sq_deviation)
            
        for i in range(n_draws):
            nu = math.exp(random.uniform(nu_r_draw_range[0], nu_r_draw_range[1]))
            r = math.exp(random.uniform(nu_r_draw_range[0], nu_r_draw_range[1]))
            mu = random.uniform(mu_draw_range[0], mu_draw_range[1])
            s = random.uniform(s_draw_range[0], s_draw_range[1])
            
            this_draw = dict(nu=nu, r=r, mu=mu, s=s)
            
            samples.append(this_draw)
            
        assert len(samples) == n_draws
        
        return samples
    
    
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
        params = ['mu', 'nu', 'r', 's']
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
            
        