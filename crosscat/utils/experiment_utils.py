"""A framework for running engine-agnostic experiments.

The experiment engine must provide 'runner', 'reader', 'writer' functions.  The
'runner' converts 'config's into 'result's.  'writer' and 'reader' serialize and
deserialize 'result's.

A 'result' must be a dictionary with at least one key, 'config', which includes
the actual config used to generate the 'result'.

fs_* functions provide template examples of writer, reader, and other support
functions that use the local filesystem for persistent storage.

"""

import os
import operator
import functools
#
from crosscat.utils.file_utils import pickle, unpickle, ensure_dir
from crosscat.utils.general_utils import ensure_listlike


####################################
# file system storage implementation
def fs_write_result(config_to_filepath, result, dirname='./'):
    """Write a result to the local filesystem

    File written depends on output of config_to_filepath AND directory

    Args:
        config_to_filepath: ('config' -> string) function to generate location
            in local filesystem to write to.  Possibly modified by 'dirname'
            argument
        result: 'result' to be written to local filesystem.  Must be
            serializable via pickle
        dirname: (string) directory to prepend to output of config_to_filepath

    Returns:
        None
    """

    config = result['config']
    filepath = config_to_filepath(config)
    # filepath may contain directories
    full_filepath = os.path.join(dirname, filepath)
    _dirname = os.path.split(full_filepath)[0]
    ensure_dir(_dirname)
    #
    pickle(result, filepath, dir=dirname)
    return

def fs_read_result(config_to_filepath, config, dirname='./'):
    """Read a result from the local filesystem

    File read depends on output of config_to_filepath AND dirname

    Args:
        config_to_filepath: ('config' -> string) function to generate location
            in local filesystem to read from.  Possibly modified by 'dirname'
            argument
        config: 'config' used to generate filepath
        dirname: (string) directory to prepend to output of config_to_filepath

    Returns:
        None
    """

    filepath = config_to_filepath(config)
    result = unpickle(filepath, dirname)
    return result

def fs_find_results(is_result_filepath, dirname='./'):
    """Searches a directory for files that contain 'config's

    Looks ONLY in the specified directory, not recursively.

    Args:
        is_result_filepath: (string -> bool) function to determine if reading
            filepath would generate a 'result'
        dirname: (string) local filesystem directory to look in

    Returns:
        filepaths: (list of strings) list of filepaths that could be passed to
        'open'
    """

    def get_result_files((root, directories, filenames)):
        my_join = lambda filename: os.path.join(root, filename)
        filepaths = map(my_join, filenames)
        filepaths = filter(is_result_filepath, filepaths)
        return filepaths
    def is_this_dirname(filepath):
        _dir, _file = os.path.split(filepath)
        return os.path.split(_dir)[0] == dirname
    filepaths_list = map(get_result_files, os.walk(dirname))
    filepaths = reduce(operator.add, filepaths_list)
    filepaths = filter(is_this_dirname, filepaths)
    return filepaths

def fs_read_all_configs(is_result_filepath, dirname='./'):
    """Reads and extracts 'config's from all files that contain 'config's in a
    directory

    Args:
        is_result_filepath: (string -> bool) function to determine if reading
            filepath would generate a 'result'
        dirname: (string) local filesystem directory to look in

    Returns:
        config_list: (list of 'config's) list of all 'config's found

    """

    def _read_config(filepath):
        result = unpickle(filepath)
        config = result['config']
        return config
    filepaths = fs_find_results(is_result_filepath, dirname)
    config_list = map(_read_config, filepaths)
    return config_list


################
# core functions
def read_results(reader, config_list, dirname='./'):
    """Reads and extracts 'result's from all files that contain 'result's in a
    directory

    Args:
        config_list: (list of 'config's) list of 'config's to read using reader
        reader: ('config', string -> 'result') function to read 'result' from
            persistent storage.
        given a
            'config' and a dirname
        dirname: (string) local filesystem directory to look in

    Returns:
        results: (list of 'result's) list of all 'result's found

    """

    config_list = ensure_listlike(config_list)
    _read_result = functools.partial(reader, dirname=dirname)
    results = map(_read_result, config_list)
    return results

def do_experiment(config, runner, writer, dirname='./'):
    """Runs and writes provided 'config' using provided runner, writer

    Args:
        config: ('config') 'config' to run with runner
        runner: ('config' -> 'result') function that takes config and returns
            result.  This is where the computation occurs.
        writer: ('result', string -> None) function that takes single result and writes
            to some persistent storage
        dirname: (string) directory to write serialized 'result's to

    Returns:
        None
    """

    result = runner(config)
    writer(result, dirname)
    return

def do_experiments(config_list, runner, writer, dirname='./', mapper=map):
    """Runs and writes provided 'config's using provided runner, writer, mapper

    Same as do_experiment but takes list of 'config's and optional mapper
    argument.  Optional mapper argument allows multiprocessing or
    IPython.parallel

    Args:
        config_list: (list of 'config's) 'config's to run with runner
        runner: ('config' -> 'result') function that takes config and returns
            result.  This is where the computation occurs.
        writer: ('result' -> None) function that takes single result and writes
            it to local filesystem
        dirname: (string) local filesystem directory to write serialize
            'result's to
        mapper: (function, args -> outputs) mapper to use.  If runner spawns
        daemonic processes, mapper must be non-daemonic.

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
    import crosscat.utils.geweke_utils as geweke_utils
    from crosscat.utils.general_utils import MapperContext, NoDaemonPool

    # demonstrate using geweke_utils
    is_result_filepath = geweke_utils.is_summary_file
    config_to_filepath = geweke_utils.config_to_filepath
    runner = geweke_utils.run_geweke
    #
    args_to_config = geweke_utils.args_to_config

    # use provided local file system writer, reader
    writer = functools.partial(fs_write_result, config_to_filepath)
    reader = functools.partial(fs_read_result, config_to_filepath)
    read_all_configs = functools.partial(fs_read_all_configs,
            is_result_filepath)

    # experiment settings
    args_list = [
            ['--num_rows', '10', '--num_cols', '2', '--num_iters', '300', ],
            ['--num_rows', '10', '--num_cols', '3', '--num_iters', '300', ],
            ['--num_rows', '20', '--num_cols', '2', '--num_iters', '300', ],
            ['--num_rows', '20', '--num_cols', '3', '--num_iters', '300', ],
            ]
    dirname = 'my_expt_bank'
    config_list = map(args_to_config, args_list)

    # run experiments
    with MapperContext(Pool=NoDaemonPool) as mapper:
        # use non-daemonic mapper since run_geweke spawns daemonic processes
        do_experiments(config_list, runner, writer, dirname, mapper)

    # demonstrate reading experiments
    configs_list = read_all_configs(dirname)
    has_three_cols = lambda config: config['num_cols'] == 3
    configs_list = filter(has_three_cols, configs_list)
    results = read_results(reader, configs_list, dirname)
