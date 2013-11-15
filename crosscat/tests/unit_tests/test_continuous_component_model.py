import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import random
import math
import numpy

import unittest

class TestContunuousComponentModelExtensions_FromParametersConstructor(unittest.TestCase):

    def setUp(self):
        N = 10
        random.seed(0)
        self.X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])
        self.component_model = ccmext.p_ContinuousComponentModel.from_parameters(N,gen_seed=0)

    def test_all_hyperparameters_intialized(self):  
        these_hyperparameters = self.component_model.get_hypers()
        # make sure each key exists
        for hyperparameter in ['mu', 'nu', 'r', 's']:
            assert(hyperparameter in these_hyperparameters.keys())

    def test_all_suffstats_intialized(self):
        these_suffstats = self.component_model.get_suffstats()
        # make sure each key exists
        for suffstat in ['sum_x', 'sum_x_squared']:
            assert suffstat in these_suffstats.keys()

    def test_draw_component_model_params(self):
        draw = self.component_model.sample_parameters_given_hyper()
        
        assert type(draw) is dict
        
        model_parameter_bounds = self.component_model.get_model_parameter_bounds()
        
        for key, value in draw.iteritems():
            assert(key in ['mu', 'rho'])
            assert(type(value) is float or type(value) is numpy.float64)
            assert(not math.isnan(value))
            assert(not math.isinf(value))
            if key == 'rho':
                assert(value > 0.0)
    
    def test_uncollapsed_likelihood(self):
        ans = -14.248338610116935
        log_likelihood = self.component_model.uncollapsed_likelihood(self.X, {'mu':0.0, 'rho':1.0})
        assert log_likelihood < 0.0 
        assert math.fabs(ans-log_likelihood) < .00000001


class TestContunuousComponentModelExtensions_FromDataConstructor(unittest.TestCase):

    def setUp(self):
        N = 10
        random.seed(0)
        self.X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])
        self.component_model = ccmext.p_ContinuousComponentModel.from_data(self.X,gen_seed=0)

    def test_all_hyperparameters_intialized(self):  
        these_hyperparameters = self.component_model.get_hypers()
        # make sure each key exists
        for hyperparameter in ['mu', 'nu', 'r', 's']:
            assert(hyperparameter in these_hyperparameters.keys())

    def test_all_suffstats_intialized(self):
        these_suffstats = self.component_model.get_suffstats()
        # make sure each key exists
        for suffstat in ['sum_x', 'sum_x_squared']:
            assert suffstat in these_suffstats.keys()

    def test_draw_component_model_params(self):
        draw = self.component_model.sample_parameters_given_hyper()
        
        assert type(draw) is dict
                
        for key, value in draw.iteritems():
            assert(key in ['mu', 'rho'])
            assert(type(value) is float or type(value) is numpy.float64)
            assert(not math.isnan(value))
            assert(not math.isinf(value))
            if key == 'rho':
                assert(value > 0.0)

    def test_uncollapsed_likelihood(self):
        ans = -20.971295328329504
        log_likelihood = self.component_model.uncollapsed_likelihood(self.X, {'mu':0.0, 'rho':1.0})
        assert log_likelihood < 0.0 
        assert math.fabs(ans-log_likelihood) < .00000001

class TestContunuousComponentModelExtensions_static(unittest.TestCase):
    def setUp(self):
        N = 10
        random.seed(0)
        self.X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])
        self.component_class = ccmext.p_ContinuousComponentModel

    def test_log_likelihood(self):
        X_1 = numpy.array([[1],[0]])
        parameters = dict(mu=0.0, rho=1.0)
        log_likelihood = self.component_class.log_likelihood(X_1, parameters)
        assert log_likelihood < 0.0 
        assert math.fabs(-2.3378770664093453-log_likelihood) < .00000001

        parameters = dict(mu=2.2, rho=12.1)
        log_likelihood = self.component_class.log_likelihood(X_1, parameters)
        assert log_likelihood < 0.0 
        assert math.fabs(-37.338671613806667-log_likelihood) < .00000001

    def test_log_pdf(self):
        # test some answers
        X_1 = numpy.array([[1],[0]])
        parameters = dict(mu=0.0, rho=1.0)
        log_pdf = self.component_class.log_pdf(X_1, parameters)
        assert len(log_pdf) == 2
        assert math.fabs(-1.4189385332046727-log_pdf[0,0]) < .00000001
        assert math.fabs(-0.91893853320467267-log_pdf[1,0]) < .00000001

        parameters = dict(mu=2.2, rho=12.1)
        log_pdf = self.component_class.log_pdf(X_1, parameters)
        assert len(log_pdf) == 2
        assert math.fabs(-8.38433580690333-log_pdf[0,0]) < .00000001
        assert math.fabs(-28.954335806903334-log_pdf[1,0]) < .00000001

        # points that are farther away from the mean should be less likely
        parameters = dict(mu=0.0, rho=1.0)
        lspc = numpy.linspace(0,10,num=20)
        X_2 = numpy.array([[x] for x in lspc])
        log_pdf = self.component_class.log_pdf(X_2, parameters)
        assert len(log_pdf) == 20
        for n in range(1,20):
            assert log_pdf[n-1,0] > log_pdf[n,0]

    def test_generate_discrete_support(self):
        parameters = dict(mu=0.0, rho=1.0)

        support = self.component_class.generate_discrete_support(parameters, support=0.95, nbins=100)

        assert type(support) is list
        assert len(support) == 100
        # end points should have the same magnitude
        assert support[0] == -support[-1] 
        # the two points stradding the mean should have the same magnitude
        assert support[49] == -support[50]
        assert math.fabs(support[0] + 1.959963984540054) < .00000001
        assert math.fabs(support[-1] - 1.959963984540054) < .00000001

    def test_draw_component_model_hyperparameters_single(self):
        draw_list = self.component_class.draw_hyperparameters(self.X)
        assert type(draw_list) is list
        assert type(draw_list[0]) is dict

        draw = draw_list[0]

        assert type(draw) is dict
        
        for key, value in draw.iteritems():
            assert key in ['mu', 'nu', 'r', 's']
            assert type(value) is float or type(value) is numpy.float64
            assert(not math.isnan(value))
            assert(not math.isinf(value))

            if key in ['nu', 's', 'r']:
                assert value > 0.0

    def test_draw_component_model_hyperparameters_multiple(self):
        n_draws = 3
        draw_list = self.component_class.draw_hyperparameters(self.X, n_draws=n_draws)

        assert type(draw_list) is list
        assert len(draw_list) == 3

        for draw in draw_list:
            assert type(draw) is dict
        
            for key, value in draw.iteritems():
                assert key in ['mu', 'nu', 'r', 's']
                assert type(value) is float or type(value) is numpy.float64
                assert(not math.isnan(value))
                assert(not math.isinf(value))

                if key in ['nu', 's', 'r']:
                    assert value > 0.0

    def test_generate_data_from_parameters(self):
        N = 10
        parameters = dict(mu=0.0, rho=1.0)
        X = self.component_class.generate_data_from_parameters(parameters, N)

        assert type(X) == numpy.ndarray
        assert len(X) == N



if __name__ == '__main__':
    unittest.main()
