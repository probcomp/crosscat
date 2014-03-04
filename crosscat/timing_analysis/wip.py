import itertools
#
from crosscat.LocalEngine import LocalEngine
import crosscat.utils.data_utils as du
import crosscat.utils.timing_test_utils as ttu


base_config = dict(
        gen_seed=0, inf_seed=0,
        num_rows=10, num_cols=10, num_clusters=1, num_views=1,
        n_steps=10,
        )

def gen_config(**kwargs):
    config = base_config.copy()
    config.update(kwargs)
    return config

def gen_configs(**kwargs):
    keys = kwargs.keys()
    values_lists = kwargs.values()
    make_dict = lambda values: dict(zip(keys, values))
    kwargs_list = map(make_dict, itertools.product(*values_lists))
    configs = [gen_config(**_kwargs) for _kwargs in kwargs_list]
    return configs

def _munge_config(config):
    generate_args = config.copy()
    generate_args['num_splits'] = generate_args.pop('num_views')
    #
    analyze_args = dict()
    analyze_args['n_steps'] = generate_args.pop('n_steps')
    analyze_args['kernel_list'] = generate_args.pop('kernel_list')
    #
    inf_seed = generate_args.pop('inf_seed')
    return generate_args, analyze_args, inf_seed

def runner(config):
    generate_args, analyze_args, inf_seed = _munge_config(config)
    # generate synthetic data
    T, M_c, M_r, X_L, X_D = ttu.generate_clean_state(max_mean=10, max_std=1,
            **generate_args)
    table_shape = map(len, (T, T[0]))
    start_dims = du.get_state_shape(X_L)
    # run engine with do_timing = True
    engine = LocalEngine(inf_seed)
    X_L, X_D, (elapsed_secs,) = engine.analyze(M_c, T, X_L, X_D,
            do_timing=True,
            **analyze_args
            )
    #
    end_dims = du.get_state_shape(X_L)
    ret_dict = dict(
        table_shape=table_shape,
        start_dims=start_dims,
        end_dims=end_dims,
        elapsed_secs=elapsed_secs,
        )
    return ret_dict

all_kernels = [
    'column_partition_hyperparameter',
    'column_partition_assignments',
    'column_hyperparameters',
    'row_partition_hyperparameters',
    'row_partition_assignments',
    ]


if __name__ == '__main__':
    configs = gen_configs(
            kernel_list=[[kernel] for kernel in all_kernels],
            num_rows=[10, 100],
            )
    results = map(runner, configs)
    for el in zip(configs, results):
        print; print el; print

