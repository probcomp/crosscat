#
#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
#   Research Leads: Vikash Mansinghka, Patrick Shafto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
import itertools
#
import numpy
#
import crosscat.utils.data_utils as du
import crosscat.utils.xnet_utils as xu
from crosscat.LocalEngine import LocalEngine
import crosscat.cython_code.State as State


def get_generative_clustering(M_c, M_r, T,
                              data_inverse_permutation_indices,
                              num_clusters, num_views):
    # NOTE: this function only works because State.p_State doesn't use
    #       column_component_suffstats
    num_rows = len(T)
    num_cols = len(T[0])
    X_D_helper = numpy.repeat(range(num_clusters), (num_rows / num_clusters))
    gen_X_D = [
        X_D_helper[numpy.argsort(data_inverse_permutation_index)]
        for data_inverse_permutation_index in data_inverse_permutation_indices
        ]
    gen_X_L_assignments = numpy.repeat(range(num_views), (num_cols / num_views))
    # initialize to generate an X_L to manipulate
    local_engine = LocalEngine()
    bad_X_L, bad_X_D = local_engine.initialize(M_c, M_r, T,
                                                         initialization='apart')
    bad_X_L['column_partition']['assignments'] = gen_X_L_assignments
    # manually constrcut state in in generative configuration
    state = State.p_State(M_c, T, bad_X_L, gen_X_D)
    gen_X_L = state.get_X_L()
    gen_X_D = state.get_X_D()
    # run inference on hyperparameters to leave them in a reasonable state
    kernel_list = (
        'row_partition_hyperparameters',
        'column_hyperparameters',
        'column_partition_hyperparameter',
        )
    gen_X_L, gen_X_D = local_engine.analyze(M_c, T, gen_X_L, gen_X_D, n_steps=1,
                                            kernel_list=kernel_list)
    #
    return gen_X_L, gen_X_D

def generate_clean_state(gen_seed, num_clusters,
                         num_cols, num_rows, num_splits,
                         max_mean=10, max_std=1,
                         plot=False):
    # generate the data
    T, M_r, M_c, data_inverse_permutation_indices = \
        du.gen_factorial_data_objects(gen_seed, num_clusters,
                                      num_cols, num_rows, num_splits,
                                      max_mean=10, max_std=1,
                                      send_data_inverse_permutation_indices=True)
    # recover generative clustering
    X_L, X_D = get_generative_clustering(M_c, M_r, T,
                                         data_inverse_permutation_indices,
                                         num_clusters, num_splits)
    return T, M_c, M_r, X_L, X_D

def generate_hadoop_dicts(which_kernels, X_L, X_D, args_dict):
    for which_kernel in which_kernels:
        kernel_list = (which_kernel, )
        dict_to_write = dict(X_L=X_L, X_D=X_D)
        dict_to_write.update(args_dict)
        # must write kernel_list after update
        dict_to_write['kernel_list'] = kernel_list
        yield dict_to_write

def write_hadoop_input(input_filename, X_L, X_D, n_steps, SEED):
    # prep settings dictionary
    time_analyze_args_dict = xu.default_analyze_args_dict
    time_analyze_args_dict['command'] = 'time_analyze'
    time_analyze_args_dict['SEED'] = SEED
    time_analyze_args_dict['n_steps'] = n_steps
    # one kernel per line
    all_kernels = State.transition_name_to_method_name_and_args.keys()
    n_tasks = 0
    with open(input_filename, 'w') as out_fh:
        dict_generator = generate_hadoop_dicts(all_kernels, X_L, X_D, time_analyze_args_dict)
        for dict_to_write in dict_generator:
            xu.write_hadoop_line(out_fh, key=dict_to_write['SEED'], dict_to_write=dict_to_write)
            n_tasks += 1
    return n_tasks

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
    T, M_c, M_r, X_L, X_D = generate_clean_state(max_mean=10, max_std=1,
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
        config=config,
        table_shape=table_shape,
        start_dims=start_dims,
        end_dims=end_dims,
        elapsed_secs=elapsed_secs,
        )
    return ret_dict
