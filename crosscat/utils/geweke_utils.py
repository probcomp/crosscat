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
import matplotlib
matplotlib.use('Agg')
#
import multiprocessing
import collections
import functools
import operator
import re
import os
import argparse
#
import numpy
import pylab
#
import crosscat.LocalEngine as LE
import crosscat.utils.general_utils as gu
import crosscat.utils.data_utils as du
import crosscat.utils.plot_utils as pu
import crosscat.tests.quality_tests.quality_test_utils as qtu
import experiment_runner.experiment_utils as eu


image_format = 'png'
default_n_grid=31
dirname_prefix='geweke_on_schemas'
result_filename = 'result.pkl'


def sample_T(engine, M_c, T, X_L, X_D):
    row_indices = range(len(T))
    generated_T, T, X_L, X_D = engine.sample_and_insert(M_c, T, X_L, X_D,
            row_indices)
    return generated_T

def collect_diagnostics(X_L, diagnostics_data, diagnostics_funcs):
    for key, func in diagnostics_funcs.iteritems():
        diagnostics_data[key].append(func(X_L))
    return diagnostics_data

def generate_diagnostics_funcs_for_column(X_L, column_idx):
    discard_keys = ['fixed', 'K']
    keys = set(X_L['column_hypers'][column_idx].keys())
    keys = keys.difference(discard_keys)
    def helper(column_idx, key):
        func_name = 'col_%s_%s' % (column_idx, key)
        func = lambda X_L: X_L['column_hypers'][column_idx][key]
        return func_name, func
    diagnostics_funcs = { helper(column_idx, key) for key in keys }
    return diagnostics_funcs

def run_posterior_chain_iter(engine, M_c, T, X_L, X_D, diagnostics_data,
        diagnostics_funcs,
        ROW_CRP_ALPHA_GRID,
        COLUMN_CRP_ALPHA_GRID,
        S_GRID, MU_GRID,
        N_GRID,
        CT_KERNEL
        ):
    X_L, X_D = engine.analyze(M_c, T, X_L, X_D,
                S_GRID=S_GRID,
                ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
                COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
                MU_GRID=MU_GRID,
                N_GRID=N_GRID,
                CT_KERNEL=CT_KERNEL,
                )
    diagnostics_data = collect_diagnostics(X_L, diagnostics_data,
            diagnostics_funcs)
    T = sample_T(engine, M_c, T, X_L, X_D)
    return M_c, T, X_L, X_D

def arbitrate_plot_rand_idx(plot_rand_idx, num_iters):
    if plot_rand_idx is not None:
        if type(plot_rand_idx) == bool:
            if plot_rand_idx:
                plot_rand_idx = numpy.random.randint(num_iters)
            else:
                plot_rand_idx = None
                pass
            pass
        pass
    return plot_rand_idx

get_column_crp_alpha = lambda X_L: X_L['column_partition']['hypers']['alpha']
get_view_0_crp_alpha = lambda X_L: X_L['view_state'][0]['row_partition_model']['hypers']['alpha']
default_diagnostics_funcs = dict(
        column_crp_alpha=get_column_crp_alpha,
        view_0_crp_alpha=get_view_0_crp_alpha,
        )
def generate_diagnostics_funcs(X_L, probe_columns):
    diagnostics_funcs = default_diagnostics_funcs.copy()
    for probe_column in probe_columns:
        funcs_to_add = generate_diagnostics_funcs_for_column(X_L, probe_column)
        diagnostics_funcs.update(funcs_to_add)
        pass
    return diagnostics_funcs

def run_posterior_chain(seed, M_c, T, num_iters,
        probe_columns=(0,),
        ROW_CRP_ALPHA_GRID=(), COLUMN_CRP_ALPHA_GRID=(),
        S_GRID=(), MU_GRID=(),
        N_GRID=default_n_grid,
        CT_KERNEL=0,
        plot_rand_idx=None,
        ):
    plot_rand_idx = arbitrate_plot_rand_idx(plot_rand_idx, num_iters)
    engine = LE.LocalEngine(seed)
    M_r = du.gen_M_r_from_T(T)
    X_L, X_D = engine.initialize(M_c, M_r, T, 'from_the_prior',
            ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
            COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
            S_GRID=S_GRID,
            MU_GRID=MU_GRID,
            N_GRID=N_GRID,
            )
    diagnostics_funcs = generate_diagnostics_funcs(X_L, probe_columns)
    diagnostics_data = collections.defaultdict(list)
    for idx in range(num_iters):
        M_c, T, X_L, X_D = run_posterior_chain_iter(engine, M_c, T, X_L, X_D, diagnostics_data,
                diagnostics_funcs,
                ROW_CRP_ALPHA_GRID,
                COLUMN_CRP_ALPHA_GRID,
                S_GRID, MU_GRID,
                N_GRID=N_GRID,
                CT_KERNEL=CT_KERNEL,
                )
        if idx == plot_rand_idx:
            # This DOESN'T work with multithreading
            filename = 'T_%s' % idx
            pu.plot_views(numpy.array(T), X_D, X_L, M_c, filename=filename,
                    dir='./', close=True, format=image_format)
            pass
        pass
    return diagnostics_data

def run_posterior_chains(M_c, T, num_chains, num_iters, probe_columns,
        row_crp_alpha_grid, column_crp_alpha_grid,
        s_grid, mu_grid,
        N_GRID=default_n_grid,
        CT_KERNEL=0,
        ):
    # run geweke: transition-erase loop
    helper = functools.partial(run_posterior_chain, M_c=M_c, T=T, num_iters=num_iters,
            probe_columns=probe_columns,
            ROW_CRP_ALPHA_GRID=row_crp_alpha_grid,
            COLUMN_CRP_ALPHA_GRID=column_crp_alpha_grid,
            S_GRID=s_grid,
            MU_GRID=mu_grid,
            N_GRID=N_GRID,
            CT_KERNEL=CT_KERNEL,
            # this breaks with multiprocessing
            plot_rand_idx=(num_chains==1),
            )
    seeds = range(num_chains)
    do_multiprocessing = num_chains != 1
    with gu.MapperContext(do_multiprocessing) as mapper:
        diagnostics_data_list = mapper(helper, seeds)
        pass
    return diagnostics_data_list

def _forward_sample_from_prior(inf_seed_and_n_samples, M_c, T,
        probe_columns=(0,),
        ROW_CRP_ALPHA_GRID=(), COLUMN_CRP_ALPHA_GRID=(),
        S_GRID=(), MU_GRID=(),
        N_GRID=default_n_grid,
        ):
    inf_seed, n_samples = inf_seed_and_n_samples
    T = numpy.zeros(numpy.array(T).shape).tolist()
    M_r = du.gen_M_r_from_T(T)
    engine = LE.LocalEngine(inf_seed)
    diagnostics_data = collections.defaultdict(list)
    diagnostics_funcs = None
    for sample_idx in range(n_samples):
        X_L, X_D = engine.initialize(M_c, M_r, T,
                ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
                COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
                S_GRID=S_GRID,
                MU_GRID=MU_GRID,
                N_GRID=N_GRID,
                )
        if diagnostics_funcs is None:
            diagnostics_funcs = generate_diagnostics_funcs(X_L, probe_columns)
        diagnostics_data = collect_diagnostics(X_L, diagnostics_data,
                diagnostics_funcs)
        pass
    return diagnostics_data

def forward_sample_from_prior(inf_seed, n_samples, M_c, T,
        probe_columns=(0,),
        ROW_CRP_ALPHA_GRID=(), COLUMN_CRP_ALPHA_GRID=(),
        S_GRID=(), MU_GRID=(),
        do_multiprocessing=True,
        N_GRID=default_n_grid,
        ):
    helper = functools.partial(_forward_sample_from_prior, M_c=M_c, T=T,
            probe_columns=probe_columns,
            ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
            COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
            S_GRID=S_GRID,
            MU_GRID=MU_GRID,
            N_GRID=N_GRID,
            )
    cpu_count = 1 if not do_multiprocessing else multiprocessing.cpu_count()
    with gu.MapperContext(do_multiprocessing) as mapper:
        seeds = numpy.random.randint(32676, size=cpu_count)
        n_samples_list = gu.divide_N_fairly(n_samples, cpu_count)
        forward_sample_data_list = mapper(helper, zip(seeds, n_samples_list))
        forward_sample_data = condense_diagnostics_data_list(forward_sample_data_list)
    return forward_sample_data

def condense_diagnostics_data_list(diagnostics_data_list):
    def get_key_condensed(key):
        get_key = lambda x: x.get(key)
        return reduce(operator.add, map(get_key, diagnostics_data_list))
    keys = diagnostics_data_list[0].keys()
    return { key : get_key_condensed(key) for key in keys}

def generate_bins_unique(data):
    bins = sorted(set(data))
    delta = bins[-1] - bins[-2]
    bins.append(bins[-1] + delta)
    return bins

def do_hist_labelling(variable_name):
    title_str = 'Histogram for %s' % variable_name
    pylab.title(title_str)
    pylab.xlabel(variable_name)
    pylab.ylabel('frequency')
    return

def do_log_hist_bin_unique(variable_name, diagnostics_data, new_figure=True,
        do_labelling=True,
        ):
    data = diagnostics_data[variable_name]
    bins = generate_bins_unique(data)
    if new_figure:
        pylab.figure()
    hist_ret = pylab.hist(data, bins=bins)
    if do_labelling:
        do_hist_labelling(variable_name)
    pylab.gca().set_xscale('log')
    return hist_ret

def do_hist(variable_name, diagnostics_data, n_bins=31, new_figure=True,
        do_labelling=True,
        ):
    data = diagnostics_data[variable_name]
    if new_figure:
        pylab.figure()
    pylab.hist(data, bins=n_bins)
    if do_labelling:
        do_hist_labelling(variable_name)
    return

hyper_name_mapper = dict(
        s='precision hyperparameter value',
        nu='precision hyperparameter psuedo count',
        mu='mean hyperparameter value',
        r='mean hyperparameter psuedo count',
        )
col_hyper_re = re.compile('^col_([^_]*)_(.*)$')
def map_variable_name(variable_name):
    mapped_variable_name = variable_name
    match = col_hyper_re.match(variable_name)
    if match is not None:
        column_idx, hyper_name = match.groups()
        mapped_hyper_name = hyper_name_mapper.get(hyper_name, hyper_name)
        mapped_variable_name = 'column %s %s' % (column_idx, mapped_hyper_name)
        pass
    return mapped_variable_name

plotter_lookup = collections.defaultdict(lambda: do_log_hist_bin_unique,
        col_0_s=do_hist,
        col_0_mu=do_hist,
        col_0_r=do_hist,
        col_0_nu=do_hist,
        )

def plot_diagnostic_data(forward_diagnostics_data, diagnostics_data_list,
        kl_series_list, variable_name,
        parameters=None, save_kwargs=None,
        ):
    plotter = plotter_lookup[variable_name]
    mapped_variable_name = map_variable_name(variable_name)
    which_idx = numpy.random.randint(len(diagnostics_data_list))
    diagnostics_data = diagnostics_data_list[which_idx]
    forward = forward_diagnostics_data[variable_name]
    not_forward_list = [el[variable_name] for el in diagnostics_data_list]
    pylab.figure()
    #
    pylab.subplot(311)
    pylab.title('Geweke analysis for %s' % mapped_variable_name)
    plotter(variable_name, forward_diagnostics_data, new_figure=False,
            do_labelling=False)
    pylab.ylabel('Forward samples\n mass')
    #
    pylab.subplot(312)
    plotter(variable_name, diagnostics_data, new_figure=False,
            do_labelling=False)
    pylab.ylabel('Posterior samples\n mass')
    #
    pylab.subplot(313)
    map(pylab.plot, kl_series_list)
    pylab.xlabel('iteration')
    pylab.ylabel('KL')
    # FIXME: remove, or do something "better"
    pylab.gca().set_ylim((0., 0.1))
    if parameters is not None:
        pu.show_parameters(parameters)
        pass
    if save_kwargs is not None:
        filename = variable_name + '_hist.png'
        pu.save_current_figure(filename, format=image_format, **save_kwargs)
        #
        filename = variable_name + '_pp.png'
        pylab.figure()
        for not_forward in not_forward_list:
            pp_plot(forward, not_forward, 100)
            pass
        pu.save_current_figure(filename, format=image_format, **save_kwargs)
        pass
    return

def plot_all_diagnostic_data(forward_diagnostics_data, diagnostics_data_list,
        kl_series_list_dict,
        parameters=None, save_kwargs=None,
        ):
    for variable_name in forward_diagnostics_data.keys():
        print 'plotting for variable: %s' % variable_name
        try:
            kl_series_list = kl_series_list_dict[variable_name]
            plot_diagnostic_data(forward_diagnostics_data, diagnostics_data_list,
                    kl_series_list,
                    variable_name, parameters, save_kwargs)
        except Exception, e:
            print 'Failed to plot_diagnostic_data for %s' % variable_name
            print e
            pass
    return

def make_same_length(*args):
    return zip(*zip(*args))

def get_count((values, bins)):
    return numpy.histogram(values, bins)[0]

def get_log_density_series(values, bins):
    bin_widths = numpy.diff(bins)
    #
    with gu.MapperContext() as mapper:
        counts = mapper(get_count, [(el, bins) for el in values])
        pass
    counts = numpy.vstack(counts).cumsum(axis=0)
    #
    ratios = counts / numpy.arange(1., len(counts) + 1.)[:, numpy.newaxis]
    densities = ratios / bin_widths[numpy.newaxis, :]
    log_densities = numpy.log(densities)
    return log_densities

def _get_kl(grid, true_series, inferred_series):
    kld = numpy.nan
    bad_value = -numpy.inf
    has_support = lambda series: sum(series==bad_value) == 0
    true_has_support = has_support(true_series)
    inferred_has_support = has_support(inferred_series)
    if true_has_support and inferred_has_support:
        kld = qtu.KL_divergence_arrays(grid, true_series,
                inferred_series, False)
        pass
    return kld

def _get_kl_tuple((grid, true_series, inferred_series)):
    return _get_kl(grid, true_series, inferred_series)

def get_fixed_gibbs_kl_series(forward, not_forward):
    forward, not_forward = make_same_length(forward, not_forward)
    forward, not_forward = map(numpy.array, (forward, not_forward))
    grid = numpy.array(sorted(set(forward).union(not_forward)))
    kls = numpy.repeat(numpy.nan, len(forward))
    try:
        bins = numpy.append(grid, grid[-1] + numpy.diff(grid)[-1])
        #
        log_true_series = get_log_density_series(forward, bins)
        log_inferred_series = get_log_density_series(not_forward, bins)
        arg_tuples = [
                (grid, x, y)
                for x, y in zip(log_true_series, log_inferred_series)
                ]
        with gu.MapperContext() as mapper:
            kls = mapper(_get_kl_tuple, arg_tuples)
            pass
    except Exception, e:
        # this definitley happens if len(grid) == 1; as in column crp alpha for
        # single column model
        pass
    return kls

def arbitrate_mu_s(num_rows, max_mu_grid=100, max_s_grid=None):
    if max_s_grid == -1:
        max_s_grid = (max_mu_grid ** 2.) / 3. * num_rows
    return max_mu_grid, max_s_grid

def write_parameters_to_text(parameters, filename, dirname='./'):
    full_filename = os.path.join(dirname, filename)
    text = gu.get_dict_as_text(parameters)
    with open(full_filename, 'w') as fh:
        fh.writelines(text + '\n')
        pass
    return

def gen_M_c(cctypes, num_values_list):
    num_cols = len(cctypes)
    colnames = range(num_cols)
    col_indices = range(num_cols)
    def helper(cctype, num_values):
        metadata_generator = du.metadata_generator_lookup[cctype]
        faux_data = range(num_values)
        return metadata_generator(faux_data)
    #
    name_to_idx = dict(zip(colnames, col_indices))
    idx_to_name = dict(zip(map(str, col_indices), colnames))
    column_metadata = map(helper, cctypes, num_values_list)
    M_c = dict(
        name_to_idx=name_to_idx,
        idx_to_name=idx_to_name,
        column_metadata=column_metadata,
        )
    return M_c

def pp_plot(_f, _p, nbins):
    ff, edges = numpy.histogram(_f, bins=nbins, density=True)
    fp, _ = numpy.histogram(_p, bins=edges, density=True)
    Ff = numpy.cumsum(ff*(edges[1:]-edges[:-1]))
    Fp = numpy.cumsum(fp*(edges[1:]-edges[:-1]))
    pylab.plot([0,1],[0,1],c='black', ls='--')
    pylab.plot(Ff,Fp, c='black')
    pylab.xlim([0,1])
    pylab.ylim([0,1])
    return

def generate_kl_series_list_dict(forward_diagnostics_data,
        diagnostics_data_list):
    kl_series_list_dict = dict()
    for variable_name in forward_diagnostics_data:
        forward = forward_diagnostics_data[variable_name]
        not_forward_list = [el[variable_name] for el in diagnostics_data_list]
        kl_series_list = [
                get_fixed_gibbs_kl_series(forward, not_forward)
                for not_forward in not_forward_list
                ]
        kl_series_list_dict[variable_name] = kl_series_list
        pass
    return kl_series_list_dict

def post_process(forward_diagnostics_data, diagnostics_data_list):
    get_final = lambda indexable: indexable[-1]
    #
    kl_series_list_dict = generate_kl_series_list_dict(forward_diagnostics_data,
            diagnostics_data_list)
    final_kls = {
            key : map(get_final, value)
            for key, value in kl_series_list_dict.iteritems()
            }
    summary_kls = {
            key : numpy.mean(value)
            for key, value in final_kls.iteritems()
            }
    return dict(
            kl_series_list_dict=kl_series_list_dict,
            final_kls=final_kls,
            summary_kls=summary_kls,
            )

def run_geweke(config):
    num_rows = config['num_rows']
    num_cols = config['num_cols']
    inf_seed = config['inf_seed']
    gen_seed = config['gen_seed']
    num_chains = config['num_chains']
    num_iters = config['num_iters']
    row_crp_alpha_grid = config['row_crp_alpha_grid']
    column_crp_alpha_grid = config['column_crp_alpha_grid']
    max_mu_grid = config['max_mu_grid']
    max_s_grid = config['max_s_grid']
    n_grid = config['n_grid']
    cctypes = config['cctypes']
    num_multinomial_values = config['num_multinomial_values']
    probe_columns = config['probe_columns']
    CT_KERNEL=config['CT_KERNEL']


    num_values_list = [num_multinomial_values] * num_cols
    M_c = gen_M_c(cctypes, num_values_list)
    T = numpy.random.uniform(0, 10, (num_rows, num_cols)).tolist()
    # may be an issue if this n_grid doesn't match the other grids in the c++
    mu_grid = numpy.linspace(-max_mu_grid, max_mu_grid, n_grid)
    s_grid = numpy.linspace(1, max_s_grid, n_grid)

    # run geweke: forward sample only
    with gu.Timer('generating forward samples') as timer:
        forward_diagnostics_data = forward_sample_from_prior(inf_seed,
                num_iters, M_c, T, probe_columns,
                row_crp_alpha_grid, column_crp_alpha_grid,
                s_grid, mu_grid,
                do_multiprocessing=True,
                N_GRID=n_grid,
                )
    # run geweke: transition-erase loop
    with gu.Timer('generating posterior samples') as timer:
        diagnostics_data_list = run_posterior_chains(M_c, T, num_chains, num_iters, probe_columns,
                row_crp_alpha_grid, column_crp_alpha_grid,
                s_grid, mu_grid,
                N_GRID=n_grid,
                CT_KERNEL=CT_KERNEL,
                )
    # post process data
    with gu.Timer('post prcessing data') as timer:
        processed_data = post_process(forward_diagnostics_data, diagnostics_data_list)
    result = dict(
            config=config,
            summary=processed_data['summary_kls'],
            forward_diagnostics_data=forward_diagnostics_data,
            diagnostics_data_list=diagnostics_data_list,
            processed_data=processed_data,
            )
    return result

parameters_to_show = ['num_rows', 'num_cols', 'max_mu_grid', 'max_s_grid',
    'n_grid', 'num_iters', 'num_chains', 'CT_KERNEL',]
def plot_result(result, dirname='./'):
    # extract variables
    config = result['config']
    forward_diagnostics_data = result['forward_diagnostics_data']
    diagnostics_data_list = result['diagnostics_data_list']
    processed_data = result['processed_data']
    kl_series_list_dict = processed_data['kl_series_list_dict']
    #
    _dirname = eu._generate_dirname(dirname_prefix, 10, config)
    save_kwargs = dict(dir=os.path.join(dirname, _dirname))
    get_tuple = lambda parameter: (parameter, config[parameter])
    parameters = dict(map(get_tuple, parameters_to_show))
    if 'cctypes' in config:
        # FIXME: remove this kludgy if statement
        counter = collections.Counter(config['cctypes'])
        parameters['Counter(cctypes)'] = dict(counter.items())
        pass
    #
    plot_all_diagnostic_data(
            forward_diagnostics_data, diagnostics_data_list,
            kl_series_list_dict,
            parameters, save_kwargs)
    return

def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rows', default=10, type=int)
    parser.add_argument('--num_cols', default=2, type=int)
    parser.add_argument('--inf_seed', default=0, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--num_chains', default=None, type=int)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--row_crp_alpha_grid', nargs='+', default=None, type=float)
    parser.add_argument('--column_crp_alpha_grid', nargs='+', default=None, type=float)
    parser.add_argument('--_divisor', default=1., type=float)

    parser.add_argument('--max_mu_grid', default=10, type=int)
    parser.add_argument('--max_s_grid', default=100, type=int)
    parser.add_argument('--n_grid', default=31, type=int)
    parser.add_argument('--CT_KERNEL', default=0, type=int)
    parser.add_argument('--num_multinomial_values', default=2, type=int)
    parser.add_argument('--cctypes', nargs='*', default=None, type=str)
    parser.add_argument('--probe_columns', nargs='*', default=None, type=str)
    return parser

def _gen_grid(N, n_grid, _divisor=1.):
    return numpy.linspace(1., N / _divisor, n_grid).tolist()

def arbitrate_args(args):
    if args.num_chains is None:
        args.num_chains = min(4, multiprocessing.cpu_count())
    if args.probe_columns is None:
        args.probe_columns = (0, 1) if args.num_cols > 1 else (0,)
    if args.cctypes is None:
        args.cctypes = ['continuous'] + ['multinomial'] * (args.num_cols - 1)
    assert len(args.cctypes) == args.num_cols
    args.max_mu_grid, args.max_s_grid = arbitrate_mu_s(args.num_rows,
            args.max_mu_grid, args.max_s_grid)
    if args.row_crp_alpha_grid is None:
       args.row_crp_alpha_grid = _gen_grid(args.num_rows, args.n_grid,
               args._divisor)
    if args.column_crp_alpha_grid is None:
        args.column_crp_alpha_grid = _gen_grid(args.num_cols, args.n_grid,
                args._divisor)
    return args

def get_chisquare(not_forward, forward=None):
    def get_sorted_counts(values):
        get_count = lambda (value, count): count
        tuples = sorted(collections.Counter(values).items())
        return map(get_count, counts)
    args = (not_forward, forward)
    args = filter(None, args)
    args = map(get_sorted_counts, args)
    return stats.chisquare(*args)

def generate_ks_stats_list(diagnostics_data_list, forward_diagnostics_data):
    from scipy import stats
    ks_stats_list = list()
    for diagnostics_data in diagnostics_data_list:
        ks_stats = dict()
        for variable_name in diagnostics_data.keys():
            stat, p = stats.ks_2samp(diagnostics_data[variable_name],
                    forward_diagnostics_data[variable_name])
            ks_stats[variable_name] = stat, p
            pass
        ks_stats_list.append(ks_stats)
        pass
    return ks_stats_list

def generate_chi2_stats_list(diagnostics_data_list, forward_diagnostics_data):
    chi2_stats_list = list()
    for diagnostics_data in diagnostics_data_list:
        chi2_stats = dict()
        for variable_name in forward_diagnostics_data.keys():
            not_forward = diagnostics_data[variable_name]
            forward = forward_diagnostics_data[variable_name]
            #chi2 = get_chisquare(not_forward, forward)
            chi2 = get_chisquare(not_forward)
            chi2_stats[variable_name] = chi2
            pass
        chi2_stats_list.append(chi2_stats)
        pass
    return chi2_stats_list


if __name__ == '__main__':
    # parse input
    parser = generate_parser()
    args = parser.parse_args()
    args = arbitrate_args(args)
    config = args.__dict__

    # the bulk of the work
    result_dict = run_geweke(config)
    plot_result(result_dict)
    #write_result(result_dict)
