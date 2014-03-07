import argparse
import random
from functools import partial
#
import numpy
import pylab
pylab.ion()
pylab.show()
#
from crosscat.LocalEngine import LocalEngine
import crosscat.utils.data_utils as du
import crosscat.utils.geweke_utils as gu
import crosscat.utils.timing_test_utils as ttu
import crosscat.utils.convergence_test_utils as ctu


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--num_rows', default=100, type=int)
    parser.add_argument('--num_cols', default=4, type=int)
    parser.add_argument('--num_clusters', default=5, type=int)
    parser.add_argument('--num_views', default=1, type=int)
    parser.add_argument('--n_steps', default=10, type=int)
    parser.add_argument('--n_test', default=None, type=int)
    return parser

def arbitrate_args(args):
    if args.n_test is None:
        args.n_test = args.num_rows / 10
    return args

def args_to_config(args):
    parser = generate_parser()
    args = parser.parse_args(args)
    args = arbitrate_args(args)
    config = args.__dict__
    return config

def test_log_likelihood_quality_test(config):
    gen_seed = config['gen_seed']
    num_rows = config['num_rows']
    num_cols = config['num_cols']
    num_clusters = config['num_clusters']
    num_views = config['num_views']
    n_steps = config['n_steps']
    n_test = config['n_test']

    # generate data
    T, M_c, M_r, gen_X_L, gen_X_D = ttu.generate_clean_state(gen_seed, num_clusters,
            num_cols, num_rows, num_views)
    engine = LocalEngine()
    sampled_T = gu.sample_T(engine, M_c, T, gen_X_L, gen_X_D)
    T_test = random.sample(sampled_T, n_test)
    gen_data_ll = ctu.calc_mean_test_log_likelihood(M_c, T, gen_X_L, gen_X_D, T)
    gen_test_set_ll = ctu.calc_mean_test_log_likelihood(M_c, T, gen_X_L, gen_X_D, T_test)

    # run inference
    def calc_ll(T, p_State):
        log_likelihoods = map(p_State.calc_row_predictive_logp, T)
        mean_log_likelihood = numpy.mean(log_likelihoods)
        return mean_log_likelihood
    calc_data_ll = partial(calc_ll, T)
    calc_test_set_ll = partial(calc_ll, T_test)
    diagnostic_func_dict = dict(
            data_ll=calc_data_ll,
            test_set_ll=calc_test_set_ll,
            )
    X_L, X_D = engine.initialize(M_c, M_r, T)
    X_L, X_D, diagnostics_dict = engine.analyze(M_c, T, X_L, X_D,
            do_diagnostics=diagnostic_func_dict, n_steps=n_steps)

    result = dict(
            diagnostics_dict=diagnostics_dict,
            gen_data_ll=gen_data_ll,
            gen_test_set_ll=gen_test_set_ll,
            )
    return result

def plot_result(result):
    diagnostics_dict = result['diagnostics_dict']
    gen_data_ll = result['gen_data_ll']
    gen_test_set_ll = result['gen_test_set_ll']
    #
    pylab.plot(diagnostics_dict['data_ll'], 'g')
    pylab.plot(diagnostics_dict['test_set_ll'], 'r')
    pylab.axhline(gen_data_ll, color='g', linestyle='--')
    pylab.axhline(gen_test_set_ll, color='r', linestyle='--')
    return

if __name__ == '__main__':
    parser = generate_parser()
    args = parser.parse_args()
    args = arbitrate_args(args)
    config = args.__dict__

    result = test_log_likelihood_quality_test(config)
    plot_result(result)
