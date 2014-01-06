#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
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
import collections
import functools
import operator
import argparse
import os
#
import numpy
import pylab
pylab.ion()
pylab.show()
#
import crosscat.settings as S
import crosscat.utils.data_utils as du
import crosscat.utils.file_utils as fu
import crosscat.LocalEngine as LE


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--num_rows', default=10, type=int)
parser.add_argument('--num_cols', default=2, type=int)
parser.add_argument('--inf_seed', default=0, type=int)
parser.add_argument('--gen_seed', default=0, type=int)
parser.add_argument('--num_chains', default=2, type=int)
parser.add_argument('--num_transitions', default=2, type=int)
args = parser.parse_args()
#
num_rows = args.num_rows
num_cols = args.num_cols
inf_seed = args.inf_seed
gen_seed = args.gen_seed
num_chains = args.num_chains
num_transitions = args.num_transitions


def determine_Q(M_c, query_names, num_rows, impute_row=None):
    name_to_idx = M_c['name_to_idx']
    query_col_indices = [name_to_idx[colname] for colname in query_names]
    row_idx = num_rows + 1 if impute_row is None else impute_row
    Q = [(row_idx, col_idx) for col_idx in query_col_indices]
    return Q

def sample_T(engine, M_c, X_L, X_D):
    num_cols = len(X_L['column_partition']['assignments'])
    query_cols = range(num_cols)
    col_names = numpy.array([M_c['idx_to_name'][str(col_idx)] for col_idx in range(num_cols)])
    query_names = col_names[query_cols]
    generated_T = []
    for row_i in range(num_rows):
        Q = determine_Q(M_c, query_names, row_i)
        sample = engine.simple_predictive_sample(M_c, X_L, X_D, None, Q, 1)[0]
        generated_T.append(sample)
    return generated_T

def generate_and_initialize(gen_seed, inf_seed, num_rows, num_cols):
    T, inverse_permutation_indices = du.gen_factorial_data(
            gen_seed=gen_seed,
            num_clusters=1,
            num_rows=num_rows,
            num_cols=num_cols,
            num_splits=1,
    		max_mean_per_category=1,
            max_std=1)
    M_r = du.gen_M_r_from_T(T)
    M_c = du.gen_M_c_from_T(T)
    # initialze and transition chains
    engine = LE.LocalEngine(inf_seed)
    X_L, X_D = engine.initialize(M_c, M_r, T, 'from_the_prior')
    return T, M_r, M_c, X_L, X_D

get_col_0_mu = lambda X_L: X_L['column_hypers'][0]['mu']
get_col_0_nu = lambda X_L: X_L['column_hypers'][0]['nu']
get_col_0_s = lambda X_L: X_L['column_hypers'][0]['s']
get_col_0_r = lambda X_L: X_L['column_hypers'][0]['r']
get_column_crp_alpha = lambda X_L: X_L['column_partition']['hypers']['alpha']
get_view_0_crp_alpha = lambda X_L: X_L['view_state'][0]['row_partition_model']['hypers']['alpha']

diagnostics_funcs = dict(
        col_0_r=get_col_0_r,
        col_0_s=get_col_0_s,
        col_0_mu=get_col_0_mu,
        col_0_nu=get_col_0_nu,
        column_crp_alpha=get_column_crp_alpha,
        view_0_crp_alpha=get_view_0_crp_alpha,
        )
def run_geweke_iter(engine, M_c, T, X_L, X_D, diagnostics_data, diagnostics_funcs):
    X_L, X_D = engine.analyze(M_c, T, X_L, X_D)
    for key, func in diagnostics_funcs.iteritems():
        diagnostics_data[key].append(func(X_L))
        pass
    T = sample_T(engine, M_c, X_L, X_D)
    return M_c, T, X_L, X_D


max_mu_grid = 10
mu_grid = numpy.linspace(-max_mu_grid, max_mu_grid, 31)
max_s_grid = (max_mu_grid /  2.) ** 2 * num_rows
s_grid = numpy.exp(numpy.linspace(0, numpy.log(max_s_grid), 31))

# my_grid = ()
def run_geweke_no_subs((seed, num_rows, num_cols, num_iters)):
    get_col_0_mu = lambda X_L: X_L['column_hypers'][0]['mu']
    get_col_0_nu = lambda X_L: X_L['column_hypers'][0]['nu']
    get_col_0_s = lambda X_L: X_L['column_hypers'][0]['s']
    get_col_0_r = lambda X_L: X_L['column_hypers'][0]['r']
    get_column_crp_alpha = lambda X_L: X_L['column_partition']['hypers']['alpha']
    get_view_0_crp_alpha = lambda X_L: X_L['view_state'][0]['row_partition_model']['hypers']['alpha']
    #
    diagnostics_funcs = dict(
            col_0_r=get_col_0_r,
            col_0_s=get_col_0_s,
            col_0_mu=get_col_0_mu,
            col_0_nu=get_col_0_nu,
            column_crp_alpha=get_column_crp_alpha,
            view_0_crp_alpha=get_view_0_crp_alpha,
            )
    #
    engine = LE.LocalEngine(seed)
    T, inverse_permutation_indices = du.gen_factorial_data(
            gen_seed=seed,
            num_clusters=1,
            num_rows=num_rows,
            num_cols=num_cols,
            num_splits=1,
            max_mean_per_category=1,
            max_std=1)
    M_r = du.gen_M_r_from_T(T)
    M_c = du.gen_M_c_from_T(T)
    # initialze and transition chains
    engine = LE.LocalEngine(seed)
    X_L, X_D = engine.initialize(M_c, M_r, T, 'from_the_prior',
            specified_s_grid=s_grid,
            specified_mu_grid=mu_grid,
            )
    diagnostics_data = collections.defaultdict(list)
    for idx in range(num_iters):
        X_L, X_D = engine.analyze(M_c, T, X_L, X_D,
            specified_s_grid=s_grid,
            specified_mu_grid=mu_grid,
            )
        fu.pickle(dict(X_L=X_L,X_D=X_D), 'X_L_X_D.pkl.gz')
        for key, func in diagnostics_funcs.iteritems():
            diagnostics_data[key].append(func(X_L))
            pass
        # T = sample_T(engine, M_c, X_L, X_D)
        query_cols = range(num_cols)
        col_names = numpy.array([M_c['idx_to_name'][str(col_idx)] for col_idx in range(num_cols)])
        query_names = col_names[query_cols]
        generated_T = []
        for impute_row in range(num_rows):
            # Q = determine_Q(M_c, query_names, row_i)
            name_to_idx = M_c['name_to_idx']
            query_col_indices = [name_to_idx[colname] for colname in query_names]
            row_idx = num_rows + 1 if impute_row is None else impute_row
            Q = [(row_idx, col_idx) for col_idx in query_col_indices]
            #
            sample = engine.simple_predictive_sample(M_c, X_L, X_D, None, Q, 1)[0]
            generated_T.append(sample)
            pass
        T = generated_T
        if False:
            # make sure data scale doesn't get too large, else turns into inf/nan
            T = numpy.array(T)
            max_magnitude = 1E10
            T[numpy.isnan(T)] = 0
            T.clip(-max_magnitude, max_magnitude)
            T = T.tolist()
            pass
        fu.pickle(T, 'T.pkl.gz')
        #
        pass
    return diagnostics_data

def run_geweke(seed, num_rows, num_cols, num_iters, diagnostics_funcs):
    engine = LE.LocalEngine(seed)
    T, M_r, M_c, X_L, X_D = generate_and_initialize(seed, seed, num_rows, num_cols)
    diagnostics_data = collections.defaultdict(list)
    for idx in range(num_iters):
        M_c, T, X_L, X_D = run_geweke_iter(engine, M_c, T, X_L, X_D, diagnostics_data,
                diagnostics_funcs)
        pass
    return diagnostics_data

def condense_diagnostics_data_list(diagnostics_data_list):
    def get_key_condensed(key):
        get_key = lambda x: x.get(key)
        return reduce(operator.add, map(get_key, diagnostics_data_list))
    keys = diagnostics_data_list[0].keys()
    return { key : get_key_condensed(key) for key in keys}

def filter_eps(data):
    data = numpy.array(data)
    is_eps = (0 < data) & (data < 1E-100)
    return data[~is_eps]

def clip_extremes(data):
    data = numpy.array(data)
    lower, upper = numpy.percentile(data, [.5, 99.5])
    return data.clip(lower, upper)

def generate_log_bins(data, n_bins=31):
    data = filter_eps(data)
    log_min, log_max = numpy.log(min(data)), numpy.log(max(data))
    return numpy.exp(numpy.linspace(log_min, log_max, n_bins))

def generate_log_bins_unique(data):
    data = filter_eps(data)
    bins = sorted(set(data))
    delta = bins[-1] - bins[-2]
    bins.append(bins[-1] + delta)
    return bins

def do_log_hist_bin_unique(variable_name, diagnostics_data):
    data = diagnostics_data[variable_name]
    bins = generate_log_bins_unique(data)
    pylab.figure()
    hist_ret = pylab.hist(data, bins=bins)
    pylab.title(variable_name)
    pylab.gca().set_xscale('log')
    return hist_ret

def do_log_hist(variable_name, diagnostics_data, n_bins=31):
    data = diagnostics_data[variable_name]
    data = clip_extremes(data)
    pylab.figure()
    bins = generate_log_bins(data, n_bins)
    pylab.hist(data, bins=bins)
    pylab.title(variable_name)
    pylab.gca().set_xscale('log')
    return

def do_hist(variable_name, diagnostics_data, n_bins=31):
    data = diagnostics_data[variable_name]
    data = clip_extremes(data)
    pylab.figure()
    pylab.hist(data, bins=n_bins)
    pylab.title(variable_name)
    return

plotter_lookup = collections.defaultdict(lambda: do_log_hist_bin_unique,
        col_0_s=do_log_hist,
        col_0_mu=do_hist,
        )
def plot_diagnostic_data(diagnostics_data):
    for variable_name in diagnostics_data.keys():
        plotter = plotter_lookup[variable_name]
        plotter(variable_name, diagnostics_data)
        pass
    return


# settings
num_iters = 1000
num_chains = 8
seeds = range(num_chains)
import multiprocessing
mapper = multiprocessing.Pool().map
# mapper = map


# run geweke
to_map = [(seed, num_rows, num_cols, num_iters) for seed in seeds]
diagnostics_data_list = mapper(run_geweke_no_subs, to_map)
diagnostics_data = condense_diagnostics_data_list(diagnostics_data_list)
plot_diagnostic_data(diagnostics_data)

