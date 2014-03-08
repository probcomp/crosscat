import argparse


def _generate_parser():
    default_num_rows = [100, 400, 1000, 4000]
    default_num_cols = [8, 16, 32]
    default_num_clusters = [1, 2]
    default_num_views = [1, 2]
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='timing_analysis', type=str)
    parser.add_argument('--num_rows', nargs='+', default=default_num_rows, type=int)
    parser.add_argument('--num_cols', nargs='+', default=default_num_cols, type=int)
    parser.add_argument('--num_clusters', nargs='+', default=default_num_clusters, type=int)
    parser.add_argument('--num_views', nargs='+', default=default_num_views, type=int)
    parser.add_argument('--no_plots', action='store_true')
    return parser

def _munge_args(args):
    kwargs = args.__dict__.copy()
    dirname = kwargs.pop('dirname')
    generate_plots = not kwargs.pop('no_plots')
    return kwargs, dirname, generate_plots


if __name__ == '__main__':
    from crosscat.utils.general_utils import Timer, MapperContext, NoDaemonPool
    from crosscat.utils.timing_test_utils import reader, read_all_configs, \
            read_results, writer, runner, gen_configs
    import crosscat.utils.timing_test_utils as ttu
    import experiment_runner.experiment_utils as eu

    # parse args
    parser = _generate_parser()
    args = parser.parse_args()
    kwargs, dirname, generate_plots = _munge_args(args)


    config_list = ttu.gen_configs(
            kernel_list = ttu._kernel_list,
            n_steps=[10],
            **kwargs
            )
    with Timer('experiments') as timer:
        with MapperContext(Pool=NoDaemonPool) as mapper:
            # use non-daemonic mapper since run_geweke spawns daemonic processes
            eu.do_experiments(config_list, runner, writer, dirname, mapper)
            pass
        pass


    if generate_plots:
        # read the data back in
        all_configs = read_all_configs(dirname)
        _all_results = read_results(all_configs, dirname)
        is_same_shape = lambda result: result['start_dims'] == result['end_dims']
        use_results = filter(is_same_shape, _all_results)
        # add plot_prefix so plots show up at top of list of files/folders
        ttu.plot_results(use_results, plot_prefix='_', dirname=dirname)
