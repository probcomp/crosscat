import argparse
import itertools
import functools
#
import pandas
#
import crosscat.utils.geweke_utils as geweke_utils
import crosscat.utils.experiment_utils as experiment_utils
from crosscat.utils.general_utils import MapperContext, NoDaemonPool, Timer


def generate_args_list(base_num_rows, num_iters):
    num_iters_str = str(num_iters)
    base_num_rows_str = str(base_num_rows)
    col_type_list = ['continuous', 'multinomial']
    #
    base_args = ['--num_rows', base_num_rows_str, '--num_iters', num_iters_str]
    col_type_pairs = sorted(itertools.combinations(col_type_list, 2))
    args_list = []

    # single datatype
    num_cols_list = [1, 10]
    iter_over = itertools.product(col_type_list, num_cols_list)
    for col_type, num_cols in iter_over:
        args = base_args + \
                ['--num_cols', str(num_cols), '--cctypes'] + \
                [col_type] * num_cols
        args_list.append(args)
        pass
    # pairs of datatypes
    iter_over = itertools.product(col_type_pairs, num_cols_list)
    for (col_type_a, col_type_b), num_cols in iter_over:
        args = base_args + \
                ['--num_cols', str(2 * num_cols), '--cctypes'] + \
                [col_type_a] * num_cols + \
                [col_type_b] * num_cols
        args_list.append(args)
        pass
#    # individual schemas
#    num_cols = 100
#    args = ['--num_rows', '100', '--num_iters', num_iters_str, '--num_cols',
#            str(num_cols), '--cctypes'] + ['continuous'] * num_cols
#    args_list.append(args)
#    args = ['--num_rows', '100', '--num_iters', num_iters_str, '--num_cols',
#            str(num_cols), '--num_multinomial_values', '128', '--cctypes'] + \
#                    ['multinomial'] * num_cols
#    args_list.append(args)
#    num_cols = 1000
#    args = ['--num_rows', '100', '--num_iters', num_iters_str, '--num_cols',
#            str(num_cols), '--num_multinomial_values', '2', '--cctypes'] + \
#                    ['multinomial'] * num_cols
#    args_list.append(args)
    return args_list

def plot_all_results(read_all_configs, read_results, dirname='./',
        filter_func=None):
    config_list = read_all_configs(dirname)
    config_list = filter(filter_func, config_list)
    results = read_results(config_list, dirname)
    with Timer('plotting') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since plot_result spawns daemonic processes
            plotter = functools.partial(geweke_utils.plot_result,
                    directory=dirname)
            mapper(plotter, results)
            pass
        pass
    pass

def print_all_summaries(read_all_configs, read_results, dirname='./',
        filter_func=None):
    config_list = read_all_configs(dirname)
    config_list = filter(filter_func, config_list)
    results = read_results(config_list, dirname)
    for result in results:
        print
        print result['config']
        print result['summary']
        print
        pass
    return

def result_to_series(result):
    base = result['config'].copy()
    base.update(result['summary']['summary_kls'])
    return pandas.Series(base)

def results_to_frame(results):
    series_list = map(result_to_series, results)
    return pandas.DataFrame(series_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='geweke_on_schemas', type=str)
    parser.add_argument('--base_num_rows', default=40, type=int)
    parser.add_argument('--num_iters', default=400, type=int)
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()
    dirname = args.dirname
    base_num_rows = args.base_num_rows
    num_iters = args.num_iters
    generate_plots = not args.no_plots


    is_result_filepath = geweke_utils.is_summary_file
    config_to_filepath = geweke_utils.config_to_filepath
    runner = geweke_utils.run_geweke
    args_to_config = geweke_utils.args_to_config
    #
    do_experiments = experiment_utils.do_experiments
    writer = experiment_utils.get_fs_writer(config_to_filepath)
    read_all_configs, reader, read_results = experiment_utils.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)


    args_list = generate_args_list(base_num_rows, num_iters)
    config_list = map(args_to_config, args_list)
    with Timer('experiments') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since run_geweke spawns daemonic processes
            do_experiments(config_list, runner, writer, dirname, mapper)
            pass
        pass

    read_all_configs, reader, read_results = experiment_utils.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)

    if generate_plots:
        plot_all_results(read_all_configs, read_results, dirname)
