if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vary_what', type=str, default='views')
    parser.add_argument('--input_filename', type=str, default='parsed_output')
    parser.add_argument('--plot_filename', type=str, default=None)
    parser.add_argument('--dirname', default='timing_analysis', type=str)
    args = parser.parse_args()
    input_filename = args.input_filename
    vary_what = args.vary_what
    plot_filename = args.plot_filename
    dirname = args.dirname


    # set up reader infrastructure
    import crosscat.utils.timing_test_utils as ttu
    import experiment_runner.experiment_utils as experiment_utils
    is_result_filepath = ttu.is_result_filepath
    config_to_filepath = ttu.config_to_filepath
    read_all_configs, reader, read_results = experiment_utils.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)

    # read in the data
    all_configs = read_all_configs(dirname)
    all_results = read_results(all_configs, dirname)
    ttu.plot_results(all_results, vary_what, plot_filename)
