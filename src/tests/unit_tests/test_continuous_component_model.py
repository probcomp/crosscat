import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import random
import math
import numpy

import six
import unittest

def main():
    unittest.main()

class TestContinuousComponentModelExtensions_Constructors(unittest.TestCase):
    def setUp(self):
        N = 10
        self.N = N
        random.seed(0)
        self.X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])

        self.params_good = dict(rho=1.0, mu=0.0)
        self.params_empty = dict()
        self.params_missing_rho = dict(mu=0.0)
        self.params_missing_mu = dict(mu=0.0)
        self.params_not_dict = [0.0, 1.0]
        self.params_negative_rho = dict(rho=-1.0, mu=0.0)
        self.params_zero_rho = dict(rho=0.0, mu=0.0)

        self.hypers_good = dict(mu=0.0, nu=1.0, r=1.0, s=1.0)
        self.hypers_missing_mu = dict(nu=1.0, r=1.0, s=1.0)
        self.hypers_missing_nu = dict(mu=0.0, r=1.0, s=1.0)
        self.hypers_missing_r = dict(mu=0.0, nu=1.0, s=1.0)
        self.hypers_missing_s = dict(mu=0.0, nu=1.0, r=1.0)
        self.hypers_low_nu = dict(mu=0.0, nu=-1.0, r=1.0, s=1.0)
        self.hypers_low_r = dict(mu=0.0, nu=1.0, r=-1.0, s=1.0)
        self.hypers_low_s = dict(mu=0.0, nu=1.0, r=1.0, s=-1.0)
        self.hypers_not_dict = [0,1,2,3]

    # Test from_parameters conrtuctor
    def test_from_parameters_contructor_with_good_complete_params_and_hypers(self):
        m = ccmext.p_ContinuousComponentModel.from_parameters(self.N,
            data_params=self.params_good,
            hypers=self.hypers_good,
            gen_seed=0)

        assert m is not None

    def test_from_parameters_contructor_with_no_params_and_hypers(self):
        m = ccmext.p_ContinuousComponentModel.from_parameters(self.N, gen_seed=0)
        assert m is not None

    def test_from_parameters_contructor_with_bad_params_and_good_hypers(self):
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_empty,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(TypeError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_not_dict,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_missing_mu,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_missing_rho,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_negative_rho,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_zero_rho,
            hypers=self.hypers_good,
            gen_seed=0)

    def test_from_parameters_contructor_with_good_params_and_bad_hypers(self):
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_missing_mu,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_missing_nu,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_missing_r,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_missing_s,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_low_nu,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_low_r,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_low_s,
            gen_seed=0)
        self.assertRaises(TypeError, ccmext.p_ContinuousComponentModel.from_parameters, self.N,
            data_params=self.params_good,
            hypers=self.hypers_not_dict,
            gen_seed=0)


    # From data constructor
    def test_from_data_contructor_with_good_complete_hypers(self):
        m = ccmext.p_ContinuousComponentModel.from_data(self.X,
            hypers=self.hypers_good,
            gen_seed=0)
        assert m is not None

    def test_from_data_contructor_with_no_params_and_hypers(self):
        m = ccmext.p_ContinuousComponentModel.from_data(self.X,gen_seed=0)
        assert m is not None

    def test_from_data_contructor_with_bad_hypers(self):
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_missing_mu,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_missing_nu,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_missing_r,
            gen_seed=0)
        self.assertRaises(KeyError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_missing_s,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_low_nu,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_low_r,
            gen_seed=0)
        self.assertRaises(ValueError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_low_s,
            gen_seed=0)
        self.assertRaises(TypeError, ccmext.p_ContinuousComponentModel.from_data, self.X,
            hypers=self.hypers_not_dict,
            gen_seed=0)

class TestContinuousComponentModelExtensions_FromParametersConstructor(unittest.TestCase):

    def setUp(self):
        N = 10
        random.seed(0)
        self.X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])
        self.component_model = ccmext.p_ContinuousComponentModel.from_parameters(N,gen_seed=0)

    def test_all_hyperparameters_intialized(self):  
        these_hyperparameters = self.component_model.get_hypers()
        # make sure each key exists
        for hyperparameter in [b'mu', b'nu', b'r', b's']:
            assert hyperparameter in these_hyperparameters

    def test_all_suffstats_intialized(self):
        these_suffstats = self.component_model.get_suffstats()
        # make sure each key exists
        for suffstat in [b'sum_x', b'sum_x_squared']:
            assert suffstat in these_suffstats

    def test_draw_component_model_params(self):
        draw = self.component_model.sample_parameters_given_hyper()
        
        assert type(draw) is dict
        
        model_parameter_bounds = self.component_model.get_model_parameter_bounds()
        
        for key, value in six.iteritems(draw):
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


class TestContinuousComponentModelExtensions_FromDataConstructor(unittest.TestCase):

    def setUp(self):
        N = 10
        random.seed(0)
        self.X = numpy.array([[random.normalvariate(0.0, 1.0)] for i in range(N)])
        self.component_model = ccmext.p_ContinuousComponentModel.from_data(self.X,gen_seed=0)

    def test_all_hyperparameters_intialized(self):  
        these_hyperparameters = self.component_model.get_hypers()
        # make sure each key exists
        for hyperparameter in [b'mu', b'nu', b'r', b's']:
            assert hyperparameter in these_hyperparameters

    def test_all_suffstats_intialized(self):
        these_suffstats = self.component_model.get_suffstats()
        # make sure each key exists
        for suffstat in [b'sum_x', b'sum_x_squared']:
            assert suffstat in these_suffstats

    def test_draw_component_model_params(self):
        draw = self.component_model.sample_parameters_given_hyper()
        
        assert type(draw) is dict
                
        for key, value in six.iteritems(draw):
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

class TestContinuousComponentModelExtensions_static(unittest.TestCase):
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
        
        for key, value in six.iteritems(draw):
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
        
            for key, value in six.iteritems(draw):
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
    main()
