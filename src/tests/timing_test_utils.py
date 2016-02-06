#
#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
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
import os
import functools
from collections import namedtuple, defaultdict
#
import pylab
import six
#
import crosscat.utils.data_utils as du
import crosscat.utils.xnet_utils as xu
from crosscat.LocalEngine import LocalEngine
import crosscat.cython_code.State as State
import crosscat.tests.plot_utils as pu
import experiment_runner.experiment_utils as eu


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


dirname_prefix ='timing_analysis'
all_kernels = State.transition_name_to_method_name_and_args.keys()
_kernel_list = [[kernel] for kernel in all_kernels]
base_config = dict(
        gen_seed=0, inf_seed=0,
        num_rows=10, num_cols=10, num_clusters=1, num_views=1,
        kernel_list=(), n_steps=10,
        )
gen_config = functools.partial(eu.gen_config, base_config)
gen_configs = functools.partial(eu.gen_configs, base_config)


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
    T, M_c, M_r, X_L, X_D = du.generate_clean_state(max_mean=10, max_std=1,
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
    same_shape = start_dims == end_dims
    summary = dict(
        elapsed_secs=elapsed_secs,
        same_shape=same_shape,
        )
    ret_dict = dict(
        config=config,
        summary=summary,
        table_shape=table_shape,
        start_dims=start_dims,
        end_dims=end_dims,
        )
    return ret_dict


#############
# begin nasty plotting support section
get_time_per_step = lambda timing_row: float(timing_row.time_per_step)
get_num_rows = lambda timing_row: str(int(float(timing_row.num_rows)))
get_num_cols = lambda timing_row: str(int(float(timing_row.num_cols)))
get_num_views = lambda timing_row: str(int(float(timing_row.num_views)))
get_num_clusters = lambda timing_row: str(int(float(timing_row.num_clusters)))
do_strip = lambda string: string.strip()

def group_results(timing_rows, get_fixed_parameters, get_variable_parameter):
    dict_of_dicts = defaultdict(dict)
    for timing_row in timing_rows:
        fixed_parameters = get_fixed_parameters(timing_row)
        variable_parameter = get_variable_parameter(timing_row)
        dict_of_dicts[fixed_parameters][variable_parameter] = timing_row
        pass
    return dict_of_dicts

num_cols_to_color = {'2':'k', '4':'b', '8':'c', '16':'r', '32':'m', '64':'g', '128':'c', '256':'k'}
num_cols_to_marker = {'2':'s', '4':'x', '8':'*', '16':'o', '32':'v', '64':'1', '128':'*',
    '256':'s'}
num_rows_to_color = {'100':'b', '200':'g', '400':'r', '1000':'m', '4000':'y', '10000':'g'}
num_rows_to_marker = {'100':'x', '200':'*', '400':'o', '1000':'v', '4000':'1', '10000':'*'}
num_clusters_to_marker = {'1':'s', '2':'v', '4':'x', '10':'x', '20':'o', '40':'s', '50':'v'}
num_views_to_marker = {'1':'x', '2':'o', '4':'v', '8':'*'}
#
plot_parameter_lookup = dict(
    rows=dict(
        vary_what='rows',
        which_kernel='row_partition_assignments',
        get_fixed_parameters=lambda timing_row: 'Co=%s;Cl=%s;V=%s' % \
            (timing_row.num_cols, timing_row.num_clusters,
             timing_row.num_views),
        get_variable_parameter=get_num_rows,
        get_color_parameter=get_num_cols,
        color_dict=num_cols_to_color,
        color_label_prepend='#Col=',
        get_marker_parameter=get_num_clusters,
        marker_dict=num_clusters_to_marker,
        marker_label_prepend='#Clust=',
        ),
    cols=dict(
        vary_what='cols',
        which_kernel='column_partition_assignments',
        get_fixed_parameters=lambda timing_row: 'R=%s;Cl=%s;V=%s' % \
            (timing_row.num_rows, timing_row.num_clusters,
             timing_row.num_views),
        get_variable_parameter=get_num_cols,
        get_color_parameter=get_num_rows,
        color_dict=num_rows_to_color,
        color_label_prepend='#Row=',
        get_marker_parameter=get_num_clusters,
        marker_dict=num_clusters_to_marker,
        marker_label_prepend='#Clust=',
        ),
    clusters=dict(
        vary_what='clusters',
        which_kernel='row_partition_assignments',
        get_fixed_parameters=lambda timing_row: 'R=%s;Co=%s;V=%s' % \
            (timing_row.num_rows, timing_row.num_cols,
             timing_row.num_views),
        get_variable_parameter=get_num_clusters,
        get_color_parameter=get_num_rows,
        color_dict=num_rows_to_color,
        color_label_prepend='#Row=',
        get_marker_parameter=get_num_views,
        marker_dict=num_views_to_marker,
        marker_label_prepend='#View=',
        ),
    views=dict(
        vary_what='views',
        which_kernel='column_partition_assignments',
        get_fixed_parameters=lambda timing_row: 'R=%s;Co=%s;Cl=%s' % \
            (timing_row.num_rows, timing_row.num_cols,
             timing_row.num_clusters),
        get_variable_parameter=get_num_views,
        get_color_parameter=get_num_rows,
        color_dict=num_rows_to_color,
        color_label_prepend='#Row=',
        get_marker_parameter=get_num_cols,
        marker_dict=num_cols_to_marker,
        marker_label_prepend='#Col=',
        ),
    )

get_first_label_value = lambda label: label[1+label.index('='):label.index(';')]
label_cmp = lambda x, y: cmp(int(get_first_label_value(x)), int(get_first_label_value(y)))
def plot_grouped_data(dict_of_dicts, plot_parameters):
    get_color_parameter = plot_parameters['get_color_parameter']
    color_dict = plot_parameters['color_dict']
    color_label_prepend = plot_parameters['color_label_prepend']
    timing_row_to_color = lambda timing_row: \
        color_dict[get_color_parameter(timing_row)]
    get_marker_parameter = plot_parameters['get_marker_parameter']
    marker_dict = plot_parameters['marker_dict']
    marker_label_prepend = plot_parameters['marker_label_prepend']
    timing_row_to_marker = lambda timing_row: \
        marker_dict[get_marker_parameter(timing_row)]
    vary_what = plot_parameters['vary_what']
    which_kernel = plot_parameters['which_kernel']
    def plot_run_data(configuration, run_data):
        x = sorted(run_data.keys())
        _y = [run_data[el] for el in x]
        y = map(get_time_per_step, _y)
        #
        first_timing_row = run_data.values()[0]
        color = timing_row_to_color(first_timing_row)
        marker = timing_row_to_marker(first_timing_row)
        label = str(configuration)
        pylab.plot(x, y, color=color, marker=marker, label=label)
        return
    #
    fh = pylab.figure()
    for configuration, run_data in six.iteritems(dict_of_dicts):
        plot_run_data(configuration, run_data)
    #
    pylab.xlabel('# %s' % vary_what)
    pylab.ylabel('time per step (seconds)')
    pylab.title('Timing analysis for kernel: %s' % which_kernel)

    # pu.legend_outside(bbox_to_anchor=(0.5, -.1), ncol=4, label_cmp=label_cmp)
    pu.legend_outside_from_dicts(marker_dict, color_dict,
            marker_label_prepend=marker_label_prepend,
            color_label_prepend=color_label_prepend, bbox_to_anchor=(0.5, -.1),
            label_cmp=label_cmp)
    return fh

def _munge_frame(frame):
    get_first_el = lambda row: row[0]
    # modifies frame in place
    frame['which_kernel'] = frame.pop('kernel_list').map(get_first_el)
    frame['time_per_step'] = frame.pop('elapsed_secs') / frame.pop('n_steps')
    return frame

def series_to_namedtuple(series):
    # for back-converting frame to previous plotting tool format
    index = list(series.index)
    _timing_row = namedtuple('timing_row', ' '.join(index))
    return _timing_row(*[str(series[name]) for name in index])
# end nasty plotting support section
#############

def _plot_results(_results_frame, vary_what='views', plot_filename=None):
    import experiment_runner.experiment_utils as experiment_utils
    # configure parsing/plotting
    plot_parameters = plot_parameter_lookup[vary_what]
    which_kernel = plot_parameters['which_kernel']
    get_fixed_parameters = plot_parameters['get_fixed_parameters']
    get_variable_parameter = plot_parameters['get_variable_parameter']
    get_is_this_kernel = lambda timing_row: \
        timing_row.which_kernel == which_kernel

    # munge data for plotting tools
    results_frame = _results_frame[_results_frame.same_shape]
    results_frame = _munge_frame(results_frame)
    timing_series_list = [el[1] for el in results_frame.iterrows()]
    all_timing_rows = map(series_to_namedtuple, timing_series_list)

    # filter rows
    these_timing_rows = filter(get_is_this_kernel, all_timing_rows)

    # plot
    dict_of_dicts = group_results(these_timing_rows, get_fixed_parameters,
                                  get_variable_parameter)
    plot_grouped_data(dict_of_dicts, plot_parameters)
    return

def plot_results(frame, save=True, plot_prefix=None, dirname='./'):
    # generate each type of plot
    filter_join = lambda join_with, list: join_with.join(filter(None, list))
    for vary_what in ['rows', 'cols', 'clusters', 'views']:
        plot_filename = filter_join('_', [plot_prefix, 'vary', vary_what])
        _plot_results(frame, vary_what, plot_filename)
        if save:
            pu.savefig_legend_outside(plot_filename, dir=dirname)
            pass
        pass
    return

if __name__ == '__main__':
    from experiment_runner.ExperimentRunner import ExperimentRunner


    config_list = gen_configs(
            kernel_list = _kernel_list,
            num_rows=[10, 100],
            )


    dirname = 'timing_analysis'
    er = ExperimentRunner(runner, dirname_prefix=dirname)
    er.do_experiments(config_list, dirname)
    print(er.frame)

    results_dict = er.get_results(er.frame[er.frame.same_shape])
