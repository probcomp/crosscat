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
import argparse
import os
#
import numpy
#
import crosscat.settings as S
import crosscat.utils.data_utils as du
import crosscat.utils.file_utils as fu
import crosscat.LocalEngine as LE


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--num_rows', default=50, type=int)
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


# generate data
T, inverse_permutation_indices = du.gen_factorial_data(
        gen_seed=gen_seed,
        num_clusters=1,
        num_cols=num_cols,
        num_rows=num_rows,
        num_splits=1,
		max_mean_per_category=1,
        max_std=1,
        max_mean=None)
M_r = du.gen_M_r_from_T(T)
M_c = du.gen_M_c_from_T(T)
col_names = numpy.array([M_c['idx_to_name'][str(col_idx)] for col_idx in range(num_cols)])
# initialze and transition chains
engine = LE.LocalEngine(inf_seed)
X_L, X_D = engine.initialize(M_c, M_r, T, 'from_the_prior')


# actually operate witht eh data
column_crp_alphas = []
import collections
first_column_hypers = collections.defaultdict(list)
for idx in range(10000):
    X_L, X_D= engine.analyze(M_c, T, X_L, X_D)
    column_crp_alphas.append(X_L['column_partition']['hypers']['alpha'])
    for key, value in X_L['column_hypers'][0].iteritems():
        if key == 'fixed': continue
        first_column_hypers[key].append(value)
    T = sample_T(engine, M_c, X_L, X_D)

import pylab
pylab.ion()
pylab.show()

set_bins = set(['r', 'nu'])
bins = numpy.linspace(0, numpy.log(100), 10)
for key in first_column_hypers:
    pylab.figure()
    bins_i = None
    if key in set_bins:
        bins_i = bins
        pass
    pylab.hist(first_column_hypers[key], bins=bins)
    pylab.title(key)

