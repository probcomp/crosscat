import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext
import random
import math
import numpy

import six
import unittest

def main():
    unittest.main()

class TestMultinomialComponentModelExtensions_Constructors(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.K = 3

        self.data_params_good = dict(weights=[1.0/3.0, 1/3.0, 1.0/3.0])
        self.data_params_bad_sum = dict(weights=[1.0/3.0, 1/3.0, 1.0/2.0])
        self.data_params_low_k = dict(weights=[1.0/2.0, 1/2.0])
        self.data_params_empty = dict()

        self.hypers_good = dict(K=self.K, dirichlet_alpha=1.0)
        self.hypers_missing_k = dict(dirichlet_alpha=1.0)
        self.hypers_missing_alpha = dict(K=self.K)
        self.hypers_low_k = dict(K=2, dirichlet_alpha=1.0)
        self.hypers_negative_alpha = dict(K=self.K, dirichlet_alpha=-1.0)
        self.hypers_negative_k = dict(K=-self.K, dirichlet_alpha=1.0)

        self.X_good = numpy.array([0, 2, 0, 2, 2, 2, 0, 1, 0, 2])
        self.X_high_k = numpy.array([0, 2, 0, 2, 2, 2, 3, 1, 0, 2])
        # there should be nothing wrong with this (the category exists, but we 
        # never observe it in the data)
        self.X_low_k = numpy.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0])

    # Test from_parameters conrtuctor
    def test_from_parameters_contructor_with_good_complete_params_and_hypers(self):
        m = mcmext.p_MultinomialComponentModel.from_parameters(self.N,
            params=self.data_params_good,
            hypers=self.hypers_good,
            gen_seed=0)

        assert m is not None

    def test_from_parameters_contructor_with_no_params_and_hypers(self):
        mcmext.p_MultinomialComponentModel.from_parameters(self.N,gen_seed=0)

    def test_from_parameters_contructor_with_bad_params_and_good_hypers(self):
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_bad_sum,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(KeyError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_empty,
            hypers=self.hypers_good,
            gen_seed=0)
        self.assertRaises(TypeError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=[1.0/3.0, 1/3.0, 1.0/3.0],
            hypers=self.hypers_good,
            gen_seed=0)

    def test_from_parameters_contructor_with_good_params_and_bad_hypers(self):
        self.assertRaises(KeyError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_good,
            hypers=self.hypers_missing_k,
            gen_seed=0)
        self.assertRaises(KeyError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_good,
            hypers=self.hypers_missing_alpha,
            gen_seed=0)
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_good,
            hypers=self.hypers_negative_alpha,
            gen_seed=0)
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_good,
            hypers=self.hypers_negative_k,
            gen_seed=0)

    def test_from_parameters_contructor_with_mismiatched_k_in_params_and_hypers(self):
        """
        Makes sure that an error is thrown if the number of categories doesn't match up 
        in the hyperparameters and in the model parameters
        """
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_good,
            hypers=self.hypers_low_k,
            gen_seed=0)
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_parameters, self.N,
            params=self.data_params_low_k,
            hypers=self.hypers_good,
            gen_seed=0)


    # Test from_data conrtuctor
    def test_from_data_contructor_with_good_and_complete_data_and_hypers(self):
        m = mcmext.p_MultinomialComponentModel.from_data(self.X_good,
            hypers=self.hypers_good,
            gen_seed=0)

        assert m is not None

    def test_from_data_contructor_with_good_data_and_no_hypers(self):
        mcmext.p_MultinomialComponentModel.from_data(self.X_good, gen_seed=0)

    def test_from_data_contructor_with_low_k_data_and_good_hypers(self):
        mcmext.p_MultinomialComponentModel.from_data(self.X_low_k,
            hypers=self.hypers_good,
            gen_seed=0)

    def test_from_data_contructor_with_bad_data_and_good_hypers(self):
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_data, self.X_high_k,
            hypers=self.hypers_good,
            gen_seed=0)

    def test_from_data_contructor_with_good_data_and_bad_hypers(self):
        self.assertRaises(KeyError, mcmext.p_MultinomialComponentModel.from_data, self.X_high_k,
            hypers=self.hypers_missing_k,
            gen_seed=0)
        self.assertRaises(KeyError, mcmext.p_MultinomialComponentModel.from_data, self.X_high_k,
            hypers=self.hypers_missing_alpha,
            gen_seed=0)
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_data, self.X_high_k,
            hypers=self.hypers_negative_alpha,
            gen_seed=0)
        self.assertRaises(ValueError, mcmext.p_MultinomialComponentModel.from_data, self.X_high_k,
            hypers=self.hypers_negative_k,
            gen_seed=0)


class TestMultinomialComponentModelExtensions_FromParametersConstructor(unittest.TestCase):
    def setUp(self):
        N = 10
        K = 5

        self.X = [3, 4, 1, 2, 4, 0, 3, 0, 1, 2]

        data_params_good = dict(weights=[1.0/5.0]*5)
        hypers_good = dict(K=K, dirichlet_alpha=1.0)

        self.component_model = mcmext.p_MultinomialComponentModel.from_parameters(N,gen_seed=0)
        self.component_model_w_params = mcmext.p_MultinomialComponentModel.from_parameters(N,
            params=data_params_good,
            gen_seed=0)
        self.component_model_w_hypers = mcmext.p_MultinomialComponentModel.from_parameters(N,
            hypers=hypers_good,
            gen_seed=0)
        self.component_model_w_params_and_hypers = mcmext.p_MultinomialComponentModel.from_parameters(N,
            params=data_params_good,
            hypers=hypers_good,
            gen_seed=0)

        assert self.component_model is not None

    def test_all_hyperparameters_intialized(self):  
        these_hyperparameters = self.component_model.get_hypers()
        for hyperparameter in [b'K', b'dirichlet_alpha']:
            assert hyperparameter in these_hyperparameters

        these_hyperparameters = self.component_model_w_params.get_hypers()
        for hyperparameter in [b'K', b'dirichlet_alpha']:
            assert hyperparameter in these_hyperparameters

        these_hyperparameters = self.component_model_w_hypers.get_hypers()
        for hyperparameter in [b'K', b'dirichlet_alpha']:
            assert hyperparameter in these_hyperparameters

        these_hyperparameters = self.component_model_w_params_and_hypers.get_hypers()
        for hyperparameter in [b'K', b'dirichlet_alpha']:
            assert hyperparameter in these_hyperparameters

    def test_all_suffstats_intialized(self):
        # make sure each key exists (should be keys 0,..,4)
        key_key = [(u'%d' % i).encode() for i in range(5)]

        _, these_suffstats = self.component_model.get_suffstats()
        for suffstat in key_key:
            assert(suffstat in these_suffstats.keys())

        _, these_suffstats = self.component_model_w_params.get_suffstats()
        for suffstat in key_key:
            assert(suffstat in these_suffstats.keys())

        _, these_suffstats = self.component_model_w_hypers.get_suffstats()
        for suffstat in key_key:
            assert(suffstat in these_suffstats.keys())

        _, these_suffstats = self.component_model_w_params_and_hypers.get_suffstats()
        for suffstat in key_key:
            assert(suffstat in these_suffstats.keys())

    def test_draw_component_model_params(self):
        draw = self.component_model.sample_parameters_given_hyper()
        
        assert type(draw) is dict
        
        for key, value in six.iteritems(draw):
            assert key in ['weights']
            assert type(value) is list
            assert math.fabs(sum(value)-1.0) < .0000001
            for w in value:
                assert w >= 0.0
    
    def test_uncollapsed_likelihood(self):
        params = dict(weights=[1.0/5.0]*5)
        ans = -1.27764862371727
        lp = self.component_model_w_params_and_hypers.uncollapsed_likelihood(self.X, params)
        assert math.fabs(ans-lp) < .00000001
        lp = self.component_model_w_hypers.uncollapsed_likelihood(self.X, params)
        assert math.fabs(ans-lp) < .00000001

        params = dict(weights=[.1, .1, .4, .2, .2])
        ans = -2.66394298483716
        lp = self.component_model_w_params_and_hypers.uncollapsed_likelihood(self.X, params)
        assert math.fabs(ans-lp) < .00000001
        lp = self.component_model_w_hypers.uncollapsed_likelihood(self.X, params)
        assert math.fabs(ans-lp) < .00000001


class TestMultinomialComponentModelExtensions_FromDataConstructor(unittest.TestCase):
    def setUp(self):
        K = 5
        self.X = [3, 4, 1, 2, 4, 0, 3, 0, 1, 2]

        hypers_good = dict(K=K, dirichlet_alpha=1.0)

        self.component_model = mcmext.p_MultinomialComponentModel.from_data(self.X, gen_seed=0)
        self.component_model_w_hypers = mcmext.p_MultinomialComponentModel.from_data(self.X,
            hypers=hypers_good,
            gen_seed=0)

        assert self.component_model is not None

    def test_all_hyperparameters_intialized(self):  
        these_hyperparameters = self.component_model.get_hypers()
        for hyperparameter in [b'K', b'dirichlet_alpha']:
            assert hyperparameter in these_hyperparameters

        these_hyperparameters = self.component_model_w_hypers.get_hypers()
        for hyperparameter in [b'K', b'dirichlet_alpha']:
            assert hyperparameter in these_hyperparameters


    def test_all_suffstats_intialized(self):
        # make sure each key exists (should be keys 0,..,4)
        key_key = [(u'%d' % i).encode() for i in range(5)]

        _, these_suffstats = self.component_model.get_suffstats()
        for suffstat in key_key:
            assert(suffstat in these_suffstats.keys())

        _, these_suffstats = self.component_model_w_hypers.get_suffstats()
        for suffstat in key_key:
            assert(suffstat in these_suffstats.keys())

    def test_draw_component_model_params(self):
        draw = self.component_model.sample_parameters_given_hyper()
        
        assert type(draw) is dict
        
        for key, value in six.iteritems(draw):
            assert key in ['weights']
            assert type(value) is list
            assert math.fabs(sum(value)-1.0) < .0000001
            for w in value:
                assert w >= 0.0
    
    def test_uncollapsed_likelihood(self):
        params = dict(weights=[1.0/5.0]*5)
        ans = -1.27764862371727
        lp = self.component_model_w_hypers.uncollapsed_likelihood(self.X, params)
        assert math.fabs(ans-lp) < .00000001

        params = dict(weights=[.1, .1, .4, .2, .2])
        ans = -2.66394298483716
        lp = self.component_model_w_hypers.uncollapsed_likelihood(self.X, params)
        assert math.fabs(ans-lp) < .00000001
        

class TestMultinomialComponentModelExtensions_static(unittest.TestCase):
    def setUp(self):
        N = 10
        random.seed(0)
        self.X = numpy.array([3, 4, 1, 2, 4, 0, 3, 0, 1, 2])
        self.component_class = mcmext.p_MultinomialComponentModel

    def test_log_likelihood(self):
        # the answers below are the result of MATLAB: 
        # log( mnpdf( hist(X,K), weights ) )
        ans = -4.45570245406521
        weights = [.2]*5
        log_likelihood = self.component_class.log_likelihood(self.X, 
            {'weights':weights})
        assert math.fabs(log_likelihood-ans) < .0000001

        ans = -6.78200407367657
        weights = [.1, .2, .5, .1, .1]
        log_likelihood = self.component_class.log_likelihood(self.X, 
            {'weights':weights})
        assert math.fabs(log_likelihood-ans) < .0000001

    def test_generate_discrete_support(self):
        parameters = dict(weights=[1.0/3.0, 1.0/3.0, 1.0/3.0])

        support = self.component_class.generate_discrete_support(parameters)

        assert type(support) is list
        assert len(support) == 3

        for i in range(3):
            assert i == support[i]

    def test_draw_component_model_hyperparameters_single(self):
        # test with data array
        draw_list = self.component_class.draw_hyperparameters(self.X)
        assert type(draw_list) is list
        assert type(draw_list[0]) is dict

        draw = draw_list[0]

        assert type(draw) is dict

        for key, value in six.iteritems(draw):
            assert key in ['K', 'dirichlet_alpha']

            assert(not math.isnan(value))
            assert(not math.isinf(value))

            if key == 'K':
                assert type(value) is int
                assert value == max(self.X)+1
            elif key == 'dirichlet_alpha':
                assert type(value) is float or type(value) is numpy.float64
                assert value > 0.0
            else:
                raise KeyError("Ivalid model parameters key %s" % key)

        # tests with int
        K = 5
        draw_list = self.component_class.draw_hyperparameters(K)
        assert type(draw_list) is list
        assert type(draw_list[0]) is dict

        draw = draw_list[0]

        assert type(draw) is dict

        for key, value in six.iteritems(draw):
            assert key in ['K', 'dirichlet_alpha']

            assert(not math.isnan(value))
            assert(not math.isinf(value))

            if key == 'K':
                assert type(value) is int
                assert value == K
            elif key == 'dirichlet_alpha':
                assert type(value) is float or type(value) is numpy.float64
                assert value > 0.0
            else:
                raise KeyError("Ivalid model parameters key %s" % key)

    def test_draw_component_model_hyperparameters_multiple(self):
        # test with data array
        n_draws = 3
        draw_list = self.component_class.draw_hyperparameters(self.X, n_draws=n_draws)

        assert type(draw_list) is list
        assert len(draw_list) == 3

        for draw in draw_list:
            assert type(draw) is dict
        
        for key, value in six.iteritems(draw):
            assert(not math.isnan(value))
            assert(not math.isinf(value))

            if key == 'K':
                assert type(value) is int
                assert value == max(self.X)+1
            elif key == 'dirichlet_alpha':
                assert type(value) is float or type(value) is numpy.float64
                assert value > 0.0
            else:
                raise KeyError("Ivalid model parameters key %s" % key)

        # test with int
        K = 5
        draw_list = self.component_class.draw_hyperparameters(K, n_draws=n_draws)

        assert type(draw_list) is list
        assert len(draw_list) == 3

        for draw in draw_list:
            assert type(draw) is dict
        
        for key, value in six.iteritems(draw):
            assert(not math.isnan(value))
            assert(not math.isinf(value))

            if key == 'K':
                assert type(value) is int
                assert value >= 1.0
                assert value == K
            elif key == 'dirichlet_alpha':
                assert type(value) is float or type(value) is numpy.float64
                assert value > 0.0
            else:
                raise KeyError("Ivalid model parameters key %s" % key)

    def test_generate_data_from_parameters(self):
        N = 10
        parameters = dict(weights=[1.0/3.0, 1.0/3.0, 1.0/3.0])
        X = self.component_class.generate_data_from_parameters(parameters, N, gen_seed=0)

        assert type(X) is numpy.ndarray
        assert len(X) == N
        assert max(X) <= 2
        assert min(X) >= 0

if __name__ == '__main__':
    main()
