import argparse
import itertools
import functools
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


is_result_filepath = geweke_utils.is_summary_file
config_to_filepath = geweke_utils.config_to_filepath
runner = geweke_utils.run_geweke
do_experiments = experiment_utils.do_experiments
# use provided local file system writer
_writer = experiment_utils.fs_write_result
writer = functools.partial(_writer, config_to_filepath)


def print_all_summaries(filter_func=None):
    # you could read results like this
    _read_all_configs = experiment_utils.fs_read_all_configs
    _reader = experiment_utils.fs_read_result
    read_results = experiment_utils.read_results
    read_all_configs = functools.partial(_read_all_configs, is_result_filepath)
    reader = functools.partial(_reader, config_to_filepath)
    config_list = read_all_configs(dirname)
    config_list = filter(filter_func, config_list)
    results = read_results(reader, config_list, dirname)
    for result in results:
        print
        print result['config']
        print result['summary']
        print
        pass
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='geweke_on_schemas', type=str)
    parser.add_argument('--base_num_rows', default=40, type=int)
    parser.add_argument('--num_iters', default=400, type=int)
    args = parser.parse_args()
    dirname = args.dirname
    base_num_rows = args.base_num_rows
    num_iters = args.num_iters


    args_to_config = geweke_utils.args_to_config
    args_list = generate_args_list(base_num_rows, num_iters)
    config_list = map(args_to_config, args_list)

    with Timer('experiments') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since run_geweke spawns daemonic processes
            do_experiments(config_list, runner, writer, dirname, mapper)
            pass
        pass

    # print_all_summaries()
