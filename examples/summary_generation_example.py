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
import pylab
import numpy
#
import crosscat.settings as S
import crosscat.utils.data_utils as du
import crosscat.utils.file_utils as fu
import crosscat.LocalEngine as LE


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--inf_seed', default=0, type=int)
parser.add_argument('--gen_seed', default=0, type=int)
parser.add_argument('--num_chains', default=25, type=int)
parser.add_argument('--num_transitions', default=200, type=int)
args = parser.parse_args()
#
filename = args.filename
inf_seed = args.inf_seed
gen_seed = args.gen_seed
num_chains = args.num_chains
num_transitions = args.num_transitions
#
pkl_filename = 'dha_example_num_transitions_%s.pkl.gz' % num_transitions


def determine_Q(M_c, query_names, num_rows, impute_row=None):
    name_to_idx = M_c['name_to_idx']
    query_col_indices = [name_to_idx[colname] for colname in query_names]
    row_idx = num_rows + 1 if impute_row is None else impute_row
    Q = [(row_idx, col_idx) for col_idx in query_col_indices]
    return Q

def determine_unobserved_Y(num_rows, M_c, condition_tuples):
    name_to_idx = M_c['name_to_idx']
    row_idx = num_rows + 1
    Y = []
    for col_name, col_value in condition_tuples:
        col_idx = name_to_idx[col_name]
        col_code = du.convert_value_to_code(M_c, col_idx, col_value)
        y = (row_idx, col_idx, col_code)
        Y.append(y)
    return Y

def do_initialize(seed):
    return LE._do_initialize(M_c, M_r, T, 'from_the_prior', seed)

def do_analyze(((X_L, X_D), seed)):
    return LE._do_analyze(M_c, T, X_L, X_D, (), num_transitions, (), (), -1, -1, seed)

def get_num_views(p_State):
    return len(p_State.get_X_D())

def get_marginal_logp(p_State):
    return p_State.get_marginal_logp()

def get_column_crp_alpha(p_State):
    return p_State.get_column_crp_alpha()

def get_summary_i(p_State):
    summary_funcs = [
            get_num_views,
            get_marginal_logp,
            get_column_crp_alpha,
            ]
    summary_i = [
            summary_func(p_State)
            for summary_func in summary_funcs
            ]
    return summary_i

# set everything up
T, M_r, M_c = du.read_model_data_from_csv(filename, gen_seed=gen_seed)
num_rows = len(T)
num_cols = len(T[0])
col_names = numpy.array([M_c['idx_to_name'][str(col_idx)] for col_idx in range(num_cols)])

X_L, X_D = LE._do_initialize(M_c, M_r, T, 'from_the_prior', inf_seed)
X_L, X_D, summaries = LE._do_analyze_with_summary(M_c, T, X_L, X_D, (), num_transitions,
        (), (), -1, -1, inf_seed, (get_summary_i, 1))

pylab.ion()
pylab.show()
summaries_arr = numpy.array(summaries)
num_views_vec = summaries_arr[:, 0]
marginal_logp_vec = summaries_arr[:, 1]
column_crp_alpha_vec = summaries_arr[:, 2]
#
pylab.subplot(311)
pylab.plot(num_views_vec)
pylab.xlabel('iters')
pylab.ylabel('#views')
#
pylab.subplot(312)
pylab.plot(marginal_logp_vec)
pylab.xlabel('iters')
pylab.ylabel('marginal_logp')
#
pylab.subplot(313)
pylab.plot(column_crp_alpha_vec)
pylab.xlabel('iters')
pylab.ylabel('column_crp_alpha')

