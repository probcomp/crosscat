import argparse
import operator
import itertools
from functools import partial
#
import crosscat.utils.geweke_utils as geweke_utils
import experiment_runner.experiment_utils as eu
from crosscat.utils.general_utils import MapperContext, NoDaemonPool, Timer


def _generate_args_list(num_rows, num_iters, cctypes):
    num_cols = len(cctypes)
    args_list = [
            '--num_rows', str(num_rows),
            '--num_cols', str(num_cols),
            '--num_iters', str(num_iters),
            '--cctypes'
            ] + cctypes
    return args_list

def _gen_cctypes(*args):
    _cctypes = [[cctype] * N for (cctype, N) in args]
    return reduce(operator.add, _cctypes)

def generate_args_list(base_num_rows, num_iters, do_long=False):
    num_cols_list = [1, 10]
    col_type_list = ['continuous', 'multinomial']
    col_type_pairs = sorted(itertools.combinations(col_type_list, 2))
    args_list = []

    # single datatype
    iter_over = itertools.product(col_type_list, num_cols_list)
    for col_type, num_cols in iter_over:
        cctypes = _gen_cctypes((col_type, num_cols))
        args = _generate_args_list(base_num_rows, num_iters, cctypes)
        args_list.append(args)
        pass

    # pairs of datatypes
    iter_over = itertools.product(col_type_pairs, num_cols_list)
    for (col_type_a, col_type_b), num_cols in iter_over:
        cctypes = _gen_cctypes((col_type_a, num_cols), (col_type_b, num_cols))
        args = _generate_args_list(base_num_rows, num_iters, cctypes)
        args_list.append(args)
        pass

    # hard coded runs
    if do_long:
        num_cols_long = 100
        num_rows_long = 100
        #
        cctypes = _gen_cctypes(('continuous', num_cols_long))
        args = _generate_args_list(num_rows_long, num_iters, cctypes)
        args_list.append(args)
        #
        cctypes = _gen_cctypes(('multinomial', num_cols_long))
        args = _generate_args_list(num_rows_long, num_iters, cctypes)
        args += ['--num_multinomial_values', '2']
        args_list.append(args)
        #
#        cctypes = _gen_cctypes(('multinomial', num_cols_long))
#        args = _generate_args_list(num_rows_long, num_iters, cctypes)
#        args += ['--num_multinomial_values', '128']
#        args_list.append(args)
        pass
    return args_list

def plot_results(results, dirname='./'):
    with Timer('plotting') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since plot_result spawns daemonic processes
            plotter = partial(geweke_utils.plot_result,
                    dirname=dirname)
            mapper(plotter, results)
            pass
        pass
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='geweke_on_schemas', type=str)
    parser.add_argument('--base_num_rows', default=10, type=int)
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--do_long', action='store_true')
    args = parser.parse_args()
    dirname = args.dirname
    base_num_rows = args.base_num_rows
    num_iters = args.num_iters
    do_plots = not args.no_plots
    do_long = args.do_long


    result_filename = geweke_utils.result_filename
    directory_prefix = dirname
    runner = geweke_utils.run_geweke
    arg_list_to_config = partial(eu.arg_list_to_config,
            geweke_utils.generate_parser(),
            arbitrate_args=geweke_utils.arbitrate_args)
    #
    is_result_filepath, generate_dirname, config_to_filepath = \
            eu.get_fs_helper_funcs(result_filename, directory_prefix)
    writer = eu.get_fs_writer(config_to_filepath)
    read_all_configs, reader, read_results = eu.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)

    # run experiment
    args_list = generate_args_list(base_num_rows, num_iters, do_long)
    config_list = map(arg_list_to_config, args_list)
    with Timer('experiments') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since run_geweke spawns daemonic processes
            eu.do_experiments(config_list, runner, writer, dirname, mapper)
            pass
        pass


    if do_plots:
        config_list = read_all_configs(dirname)
        results = read_results(config_list, dirname)
        plot_results(results, dirname)
        pass
