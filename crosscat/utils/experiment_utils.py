import os
import collections
#
import crosscat.utils.file_utils as file_utils
import crosscat.utils.geweke_utils as geweke_utils
import crosscat.utils.general_utils as general_utils


result_filename = geweke_utils.summary_filename


def find_configs(dirname, filename=result_filename):
    root_has_filename = lambda (root, ds, filenames): filenames.count(filename)
    get_filepath = lambda (root, ds, fs): os.path.join(root, filename)
    tuples = filter(root_has_filename, os.walk(dirname))
    filepaths = map(get_filepath, tuples)
    return filepaths

def read_all_configs(dirname='.'):
    def read_config(filepath):
        result = file_utils.unpickle(filepath, dir=dirname)
        config = result['config']
        return config
    filepaths = find_configs(dirname)
    config_list = map(read_config, filepaths)
    return config_list

def generate_filepath(config):
    _dirname = geweke_utils.generate_directory_name(**config)
    filepath = os.path.join(_dirname, result_filename)
    return filepath

def read_results(config_list, dirname=''):
    def read_result(config):
        filepath = generate_filepath(config)
        result = file_utils.unpickle(filepath, dir=dirname)
        return result
    config_list = general_utils.ensure_listlike(config_list)
    results = map(read_result, config_list)
    return results

def write_result(config, result, dirname=''):
    filepath = generate_filepath(config)
    file_utils.pickle(result, filepath, dirname=dirname)
    return

def do_experiments(runner, writer, config_list):
    def do_experiment(config):
        result = runner(**config)
        writer(config, result)
        return result
    config_list = general_utils.ensure_listlike(config_list)
    results = map(do_experiment, config_list)
    return results

if __name__ == '__main__':
    config_list = read_all_configs()
    results = read_results(config_list[:-1])

