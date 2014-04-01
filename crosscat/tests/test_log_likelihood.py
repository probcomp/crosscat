import random
import argparse
from functools import partial
#
import numpy
import pylab
#
from crosscat.LocalEngine import LocalEngine
import crosscat.utils.data_utils as du
import crosscat.utils.plot_utils as pu
import crosscat.utils.geweke_utils as gu
import crosscat.utils.convergence_test_utils as ctu
import experiment_runner.experiment_utils as eu


directory_prefix='test_log_likelihood'
#
noneify = set(['n_test'])
base_config = dict(
    gen_seed=0,
    num_rows=100, num_cols=4,
    num_clusters=5, num_views=1,
    n_steps=10, n_test=10,
    )

def runner(config):
    # helpers
    def munge_config(config):
        kwargs = config.copy()
        kwargs['num_splits'] = kwargs.pop('num_views')
        n_steps = kwargs.pop('n_steps')
        n_test = kwargs.pop('n_test')
        return kwargs, n_steps, n_test
    def calc_ll(T, p_State):
        log_likelihoods = map(p_State.calc_row_predictive_logp, T)
        mean_log_likelihood = numpy.mean(log_likelihoods)
        return mean_log_likelihood
    def gen_data(**kwargs):
        T, M_c, M_r, gen_X_L, gen_X_D = du.generate_clean_state(**kwargs)
        #
        engine = LocalEngine()
        sampled_T = gu.sample_T(engine, M_c, T, gen_X_L, gen_X_D)
        T_test = random.sample(sampled_T, n_test)
        gen_data_ll = ctu.calc_mean_test_log_likelihood(M_c, T, gen_X_L, gen_X_D, T)
        gen_test_set_ll = ctu.calc_mean_test_log_likelihood(M_c, T, gen_X_L, gen_X_D, T_test)
        #
        return T, M_c, M_r, T_test, gen_data_ll, gen_test_set_ll
    kwargs, n_steps, n_test = munge_config(config)
    T, M_c, M_r, T_test, gen_data_ll, gen_test_set_ll = gen_data(**kwargs)
    # set up to run inference
    calc_data_ll = partial(calc_ll, T)
    calc_test_set_ll = partial(calc_ll, T_test)
    diagnostic_func_dict = dict(
            data_ll=calc_data_ll,
            test_set_ll=calc_test_set_ll,
            )
    # run inference
    engine = LocalEngine()
    X_L, X_D = engine.initialize(M_c, M_r, T)
    X_L, X_D, diagnostics_dict = engine.analyze(M_c, T, X_L, X_D,
            do_diagnostics=diagnostic_func_dict, n_steps=n_steps)
    # package result
    final_data_ll = diagnostics_dict['data_ll'][-1][-1]
    final_test_set_ll = diagnostics_dict['test_set_ll'][-1][-1]
    summary = dict(
            gen_data_ll=gen_data_ll,
            gen_test_set_ll=gen_test_set_ll,
            final_data_ll=final_data_ll,
            final_test_set_ll=final_test_set_ll,
            )

    result = dict(
            config=config,
            summary=summary,
            diagnostics_dict=diagnostics_dict,
            )
    return result

def plotter(result):
    pylab.figure()
    diagnostics_dict = result['diagnostics_dict']
    gen_data_ll = result['summary']['gen_data_ll']
    gen_test_set_ll = result['summary']['gen_test_set_ll']
    #
    pylab.plot(diagnostics_dict['data_ll'], 'g')
    pylab.plot(diagnostics_dict['test_set_ll'], 'r')
    pylab.axhline(gen_data_ll, color='g', linestyle='--')
    pylab.axhline(gen_test_set_ll, color='r', linestyle='--')
    return

def _generate_parser():
    default_gen_seed = [0]
    default_num_rows = [100, 200, 500]
    default_num_cols = [10, 20]
    default_num_clusters = [1, 10, 20]
    default_num_views = [1, 5]
    default_n_steps = [10]
    default_n_test = [20]
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_seed', nargs='+', default=default_gen_seed, type=int)
    parser.add_argument('--num_rows', nargs='+', default=default_num_rows, type=int)
    parser.add_argument('--num_cols', nargs='+', default=default_num_cols, type=int)
    parser.add_argument('--num_clusters', nargs='+',
            default=default_num_clusters, type=int)
    parser.add_argument('--num_views', nargs='+', default=default_num_views,
            type=int)
    parser.add_argument('--n_steps', nargs='+', default=default_n_steps, type=int)
    parser.add_argument('--n_test', nargs='+', default=default_n_test, type=int)
    #
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--dirname', default='test_log_likelihood', type=str)
    return parser

def _munge_args(args):
    kwargs = args.__dict__.copy()
    do_plots = not kwargs.pop('no_plots')
    dirname = kwargs.pop('dirname')
    return kwargs, do_plots, dirname

def summary_plotter(results, dirname='./', save=True):
    def _scatter(x, y):
        pylab.figure()
        pylab.scatter(x, y)
        pylab.gca().set_aspect(1)
        xlim = pylab.gca().get_xlim()
        pylab.plot(xlim, xlim)
        return
    def _plot(frame, variable_suffix, filename=None, dirname='./'):
        x = frame['gen_' + variable_suffix]
        y = frame['final_' + variable_suffix]
        _scatter(x, y)
        pylab.title(variable_suffix)
        if filename is not None:
            pu.save_current_figure(filename, dir=dirname, close=True)
            pass
        return
    frame = eu.summaries_to_frame(results)
    for variable_suffix in ['test_set_ll', 'data_ll']:
        filename = variable_suffix if save else None
        _plot(frame, variable_suffix, filename=filename, dirname=dirname)
        pass
    return


if __name__ == '__main__':
    from experiment_runner.ExperimentRunner import ExperimentRunner, propagate_to_s3

    # parse args
    parser = _generate_parser()
    args = parser.parse_args()
    kwargs, do_plots, dirname = _munge_args(args)


    # create configs
    config_list = eu.gen_configs(base_config, **kwargs)


    # do experiments
    er = ExperimentRunner(runner, storage_type='fs',
            dirname_prefix=dirname,
            bucket_str='experiment_runner',
            )
    er.do_experiments(config_list)
    propagate_to_s3(er)


    if do_plots:
        results = er.get_results(er.frame).values()
        summary_plotter(results, dirname=dirname)
#        eu.plot_results(plotter, results, generate_dirname,
#                saver=pu.save_current_figure, filename='over_iters',
#                dirname=dirname)
