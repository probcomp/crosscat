import os
import operator
#
import crosscat.utils.geweke_utils as geweke_utils
from crosscat.utils.file_utils import unpickle
from crosscat.utils.general_utils import ensure_listlike


is_config_file = geweke_utils.is_summary_file
writer = geweke_utils.write_result
reader = geweke_utils.read_result
runner = geweke_utils.run_geweke


def find_configs(dirname):
    def get_config_files((root, directories, filenames)):
        join = lambda filename: os.path.join(root, filename)
        filenames = map(join, filenames)
        return filter(is_config_file, filenames)
    def is_this_dirname(filepath):
        _dir, _file = os.path.split(filepath)
        return os.path.split(_dir)[0] == dirname
    filepaths_list = map(get_config_files, os.walk(dirname))
    filepaths = reduce(operator.add, filepaths_list)
    filepaths = filter(is_this_dirname, filepaths)
    return filepaths

def read_all_configs(dirname='.'):
    def read_config(filepath):
        result = unpickle(filepath)
        config = result['config']
        return config
    filepaths = find_configs(dirname)
    config_list = map(read_config, filepaths)
    return config_list

def read_results(config_list, *args, **kwargs):
    _read_result = lambda config: reader(config, *args, **kwargs)
    config_list = ensure_listlike(config_list)
    results = map(_read_result, config_list)
    return results

def write_results(results, *args, **kwargs):
    _write_result = lambda result: writer(result, *args, **kwargs)
    map(_write_result, results)
    return

def do_experiments(runner, writer, config_list, *args, **kwargs):
    def do_experiment(config):
        result = runner(config)
        writer(result, *args, **kwargs)
        return
    config_list = ensure_listlike(config_list)
    map(do_experiment, config_list)
    return

def args_to_config(args):
    parser = geweke_utils.generate_parser()
    args = parser.parse_args(args)
    args = geweke_utils.arbitrate_args(args)
    return args.__dict__

if __name__ == '__main__':
    args_list = [
            ['--num_rows', '10', '--num_cols', '2', '--num_iters', '300', ],
            ['--num_rows', '10', '--num_cols', '3', '--num_iters', '300', ],
            ['--num_rows', '20', '--num_cols', '2', '--num_iters', '300', ],
            ['--num_rows', '20', '--num_cols', '3', '--num_iters', '300', ],
            ]
    configs_list = map(args_to_config, args_list)
    do_experiments(runner, writer, configs_list)

    configs_list = read_all_configs()
    has_three_cols = lambda config: config['num_cols'] == 3
    configs_list = filter(has_three_cols, configs_list)
    results = read_results(configs_list)
