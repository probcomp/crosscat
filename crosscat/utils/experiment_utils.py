"""A framework for running engine-agnostic experiments.

The experiment engine must provide 'runner', 'reader', 'writer' functions and
optionally a function to identify 'config' files, is_config_file.  The 'runner'
converts 'config's into 'result's.  'writer' and 'reader' serialize and
deserialize 'result's.

A 'result' must be a dictionary with at least one key: 'config' which includes
the actual config used to generate the 'result'.

is_config_file is for the special case of searching for results stored to disk.
In this case, 'reader' and 'writer' implicilty have a naming convention.

"""

import os
import operator
import functools
#
from crosscat.utils.file_utils import unpickle, ensure_dir
from crosscat.utils.general_utils import ensure_listlike, MapperContext, MyPool


def find_configs(dirname):
    """Searches a directory for files that contain 'config's

    Utilizes provided is_config_file.  Looks ONLY in the specified directory,
    not recursively

    Args:
        dirname: (string) local filesystem directory to look in

    Returns:
        filepaths: (list of strings) list of filepaths that could be passed to
        'open'
    """

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

def read_all_configs(dirname='./'):
    """Reads and extracts 'config's from all files that contain 'config's in a
    directory

    Args:
        dirname: (string) local filesystem directory to look in

    Returns:
        config_list: (list of 'config's) list of all 'config's found

    """

    def read_config(filepath):
        result = unpickle(filepath)
        config = result['config']
        return config
    filepaths = find_configs(dirname)
    config_list = map(read_config, filepaths)
    return config_list

def read_results(config_list, dirname='./'):
    """Reads and extracts 'result's from all files that contain 'result's in a
    directory

    Args:
        dirname: (string) local filesystem directory to look in

    Returns:
        results: (list of 'result's) list of all 'result's found

    """

    _read_result = lambda config: reader(config, dirname)
    config_list = ensure_listlike(config_list)
    results = map(_read_result, config_list)
    return results

def write_results(results, dirname='./'):
    """Writes all 'result's into a specified directory

    Args:
        results: (list of 'result's) list of all 'result's to write
        dirname: (string) local filesystem directory to look in

    Returns:
        None

    """

    _write_result = lambda result: writer(result, dirname)
    map(_write_result, results)
    return

def do_experiment(config, runner, writer, dirname):
    """Runs and writes provided 'config' using provided runner, writer

    Args:
        config: ('config') 'config' to run with runner
        runner: ('config' -> 'result') function that takes config and returns
            result.  This is where the computation occurs.
        writer: ('result' -> None) function that takes single result and writes
            it to local filesystem
        dirname: (string) local filesystem directory to write serialize
            'result's to

    Returns:
        None
    """

    result = runner(config)
    writer(result, dirname)
    return

def do_experiments(config_list, runner, writer, dirname='./', mapper=map):
    """Runs and writes provided 'config's using provided runner, writer, mapper

    Args:
        config_list: (list of 'config's) 'config's to run with runner
        runner: ('config' -> 'result') function that takes config and returns
            result.  This is where the computation occurs.
        writer: ('result' -> None) function that takes single result and writes
            it to local filesystem
        dirname: (string) local filesystem directory to write serialize
            'result's to
        mapper: (function, args -> outputs) mapper to use.  Enables use of
            multiprocessing or ipython.parallel

    Returns:
        None
    """

    ensure_dir(dirname)
    config_list = ensure_listlike(config_list)
    _do_experiment = functools.partial(do_experiment, runner=runner,
            writer=writer, dirname=dirname)
    mapper(_do_experiment, config_list)
    return

if __name__ == '__main__':
    # demonstrate using geweke_utils
    import crosscat.utils.geweke_utils as geweke_utils

    is_config_file = geweke_utils.is_summary_file
    writer = geweke_utils.write_result
    reader = geweke_utils.read_result
    runner = geweke_utils.run_geweke
    args_to_config = geweke_utils.args_to_config

    args_list = [
            ['--num_rows', '10', '--num_cols', '2', '--num_iters', '300', ],
            ['--num_rows', '10', '--num_cols', '3', '--num_iters', '300', ],
            ['--num_rows', '20', '--num_cols', '2', '--num_iters', '300', ],
            ['--num_rows', '20', '--num_cols', '3', '--num_iters', '300', ],
            ]
    dirname = 'my_expt_bank'

    # demonstrate generating experiments
    config_list = map(args_to_config, args_list)
    with MapperContext(Pool=MyPool) as mapper:
        do_experiments(config_list, runner, writer, dirname, mapper)

    # demonstrate reading experiments
    configs_list = read_all_configs(dirname)
    has_three_cols = lambda config: config['num_cols'] == 3
    configs_list = filter(has_three_cols, configs_list)
    results = read_results(configs_list, dirname)
