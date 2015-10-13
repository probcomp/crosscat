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
    parser.add_argument('--plot_prefix', default=None, type=str)
    parser.add_argument('--no_plots', action='store_true')
    return parser

def _munge_args(args):
    kwargs = args.__dict__.copy()
    dirname = kwargs.pop('dirname')
    plot_prefix = kwargs.pop('plot_prefix')
    generate_plots = not kwargs.pop('no_plots')
    return kwargs, dirname, plot_prefix, generate_plots


if __name__ == '__main__':
    import crosscat.tests.timing_test_utils as ttu
    from crosscat.utils.general_utils import Timer
    from experiment_runner.ExperimentRunner import ExperimentRunner, propagate_to_s3


    # parse args
    parser = _generate_parser()
    args = parser.parse_args()
    kwargs, dirname, plot_prefix, generate_plots = _munge_args(args)


    # create configs
    config_list = ttu.gen_configs(
            kernel_list = ttu._kernel_list,
            n_steps=[10],
            **kwargs
            )


    # do experiments
    er = ExperimentRunner(ttu.runner, storage_type='fs',
            dirname_prefix=dirname,
            bucket_str='experiment_runner')
    with Timer('er.do_experiments') as timer:
        er.do_experiments(config_list)
        pass
    # push to s3
    propagate_to_s3(er)


    if generate_plots:
        ttu.plot_results(er.frame, plot_prefix=plot_prefix, dirname=dirname)
