import crosscat.utils.timing_test_utils as ttu
from crosscat.utils.general_utils import Timer, MapperContext, NoDaemonPool
import experiment_runner.experiment_utils as experiment_utils


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='timing_analysis', type=str)
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()
    dirname = args.dirname
    generate_plots = not args.no_plots

    is_result_filepath = ttu.is_result_filepath
    config_to_filepath = ttu.config_to_filepath
    runner = ttu.runner
    #
    do_experiments = experiment_utils.do_experiments
    writer = experiment_utils.get_fs_writer(config_to_filepath)
    read_all_configs, reader, read_results = experiment_utils.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)


    config_list = ttu.gen_configs(
            kernel_list = ttu._kernel_list,
            num_rows=[100, 400, 1000, 4000],
            num_cols=[8, 16, 32],
            num_clusters=[1, 2],
            num_views=[1, 2],
            n_steps=[10],
            )


    with Timer('experiments') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since run_geweke spawns daemonic processes
            do_experiments(config_list, runner, writer, dirname, mapper)
            pass
        pass

    read_all_configs, reader, read_results = experiment_utils.get_fs_reader_funcs(
            is_result_filepath, config_to_filepath)

    all_configs = read_all_configs(dirname)
    _all_results = read_results(all_configs, dirname)
    is_same_shape = lambda result: result['start_dims'] == result['end_dims']
    use_results = filter(is_same_shape, _all_results)
    results_frame = experiment_utils.results_to_frame(use_results)

    if generate_plots:
        for vary_what in ['rows', 'cols', 'clusters', 'views']:
            plot_filename = 'vary_%s' % vary_what
            ttu.plot_results(use_results, vary_what, plot_filename)
            pass
        pass
