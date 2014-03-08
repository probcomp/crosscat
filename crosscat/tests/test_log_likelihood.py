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
import experiment_runner.experiment_utils as eu


result_filename = 'result.pkl'
directory_prefix='test_log_likelihood'

noneify = set(['n_test'])
base_config = dict(
    gen_seed=0,
    num_rows=100, num_cols=4,
    num_clusters=5, num_views=1,
    n_steps=10, n_test=10,
    )

def arbitrate_args(args):
    if args.n_test is None:
        args.n_test = args.num_rows / 10
    return args

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
            config=config,
            diagnostics_dict=diagnostics_dict,
            gen_data_ll=gen_data_ll,
            gen_test_set_ll=gen_test_set_ll,
            )
    return result

def plot_result(result):
    pylab.figure()
    diagnostics_dict = result['diagnostics_dict']
    gen_data_ll = result['gen_data_ll']
    gen_test_set_ll = result['gen_test_set_ll']
    #
    pylab.plot(diagnostics_dict['data_ll'], 'g')
    pylab.plot(diagnostics_dict['test_set_ll'], 'r')
    pylab.axhline(gen_data_ll, color='g', linestyle='--')
    pylab.axhline(gen_test_set_ll, color='r', linestyle='--')
    # FIXME: save the result
    return


if __name__ == '__main__':
    from crosscat.utils.general_utils import Timer, MapperContext, NoDaemonPool

    # do single experiment
    parser = eu.generate_parser(base_config, noneify)
    args = parser.parse_args()
    args = arbitrate_args(args)
    config = args.__dict__

    result = test_log_likelihood_quality_test(config)
    plot_result(result)


    # demonstrate use of experiment runner
    do_experiments = eu.do_experiments
    is_result_filepath, generate_dirname, config_to_filepath = \
            eu.get_fs_helper_funcs(result_filename, directory_prefix)
    writer = eu.get_fs_writer(config_to_filepath)
    read_all_configs, reader, read_results = eu.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)


    gen_configs_kwargs = dict(
            num_clusters=[1, 2, 4],
            num_rows = [20, 40, 100]
            )
    config_list = eu.gen_configs(base_config, **gen_configs_kwargs)
    dirname = 'test_log_likelihood'
    runner = test_log_likelihood_quality_test
    with Timer('experiments') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since run_geweke spawns daemonic processes
            do_experiments(config_list, runner, writer, dirname, mapper)
            pass
        pass


    all_configs = read_all_configs(dirname)
    all_results = read_results(all_configs, dirname)
    map(plot_result, all_results)

