import argparse
from collections import namedtuple
from collections import defaultdict
from collections import Counter
#
import pylab
#
import crosscat.utils.data_utils as du
import crosscat.utils.plot_utils as pu


get_time_per_step = lambda timing_row: float(timing_row.time_per_step)
get_num_rows = lambda timing_row: timing_row.num_rows
get_num_cols = lambda timing_row: timing_row.num_cols
get_num_views = lambda timing_row: timing_row.num_views
get_num_clusters = lambda timing_row: timing_row.num_clusters
do_strip = lambda string: string.strip()
#
def parse_timing_file(filename):
    header, rows = du.read_csv(filename)
    _timing_row = namedtuple('timing_row', ' '.join(header))
    timing_rows = []
    for row in rows:
        row = map(do_strip, row)
        timing_row = _timing_row(*row)
        timing_rows.append(timing_row)
    return timing_rows

def group_results(timing_rows, get_fixed_parameters, get_variable_parameter):
    dict_of_dicts = defaultdict(dict)
    for timing_row in timing_rows:
        # timing_row = series_to_namedtuple(timing_row)
        fixed_parameters = get_fixed_parameters(timing_row)
        variable_parameter = get_variable_parameter(timing_row)
        dict_of_dicts[fixed_parameters][variable_parameter] = timing_row
        pass
    return dict_of_dicts

num_cols_to_color = {'2':'k', '4':'b', '8':'c', '16':'r', '32':'m', '64':'g', '128':'c', '256':'k'}
num_rows_to_color = {'100':'b', '200':'g', '400':'r', '1000':'m', '4000':'y', '10000':'g'}
num_clusters_to_marker = {'1': 's', '2':'v', '10':'x', '20':'o', '40':'s', '50':'v'}
num_views_to_marker = {'1':'x', '2':'o', '4':'v'}
num_rows_to_marker = {'100':'x', '200':'*', '400':'o', '1000':'v', '4000':'1', '10000':'*'}
num_cols_to_marker = {'2':'s', '4':'x', '8':'*', '16':'o', '32':'v', '64':'1', '128':'*',
    '256':'s'}
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
def plot_grouped_data(dict_of_dicts, plot_parameters, plot_filename=None):
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
    #
    fh = pylab.figure()
    for configuration, run_data in dict_of_dicts.iteritems():
        x = sorted(run_data.keys())
        _y = [run_data[el] for el in x]
        y = map(get_time_per_step, _y)
        #
        plot_args = dict()
        first_timing_row = run_data.values()[0]
        color = timing_row_to_color(first_timing_row)
        plot_args['color'] = color
        marker = timing_row_to_marker(first_timing_row)
        plot_args['marker'] = marker
        label = str(configuration)
        plot_args['label'] = label
        #
        pylab.plot(x, y, **plot_args)
    #
    pylab.xlabel('# %s' % vary_what)
    pylab.ylabel('time per step (seconds)')
    pylab.title('Timing analysis for kernel: %s' % which_kernel)

    # pu.legend_outside(bbox_to_anchor=(0.5, -.1), ncol=4, label_cmp=label_cmp)
    pu.legend_outside_from_dicts(marker_dict, color_dict,
                                 marker_label_prepend=marker_label_prepend, color_label_prepend=color_label_prepend,
                                 bbox_to_anchor=(0.5, -.1), label_cmp=label_cmp)
                                 
                                 

    if plot_filename is not None:
        pu.savefig_legend_outside(plot_filename)
    else:
        pylab.ion()
        pylab.show()
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
    _all_results = read_results(all_configs, dirname)
    is_same_shape = lambda result: result['start_dims'] == result['end_dims']
    use_results = filter(is_same_shape, _all_results)
    results_frame = experiment_utils.results_to_frame(use_results)

    # configure parsing/plotting
    plot_parameters = plot_parameter_lookup[vary_what]
    which_kernel = plot_parameters['which_kernel']
    get_fixed_parameters = plot_parameters['get_fixed_parameters']
    get_variable_parameter = plot_parameters['get_variable_parameter']

    # munge data for plotting tools
    results_frame = _munge_frame(results_frame)
    timing_series_list = [el[1] for el in results_frame.iterrows()]
    all_timing_rows = map(series_to_namedtuple, timing_series_list)

    # filter results
    get_is_this_kernel = lambda timing_row: \
        timing_row.which_kernel == which_kernel
    these_timing_rows = filter(get_is_this_kernel, all_timing_rows)
    
    # plot
    dict_of_dicts = group_results(these_timing_rows, get_fixed_parameters,
                                  get_variable_parameter)
    plot_grouped_data(dict_of_dicts, plot_parameters, plot_filename)
