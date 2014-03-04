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
import argparse
#
import numpy
#
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State
from crosscat.JSONRPCEngine import JSONRPCEngine


parser = argparse.ArgumentParser()
parser.add_argument('--gen_seed', default=0, type=int)
parser.add_argument('--inf_seed', default=0, type=int)
parser.add_argument('--num_clusters', default=4, type=int)
parser.add_argument('--num_cols', default=16, type=int)
parser.add_argument('--num_rows', default=300, type=int)
parser.add_argument('--num_splits', default=2, type=int)
parser.add_argument('--max_mean', default=10, type=float)
parser.add_argument('--max_std', default=0.3, type=float)
parser.add_argument('--num_transitions', default=30, type=int)
parser.add_argument('--N_GRID', default=31, type=int)
parser.add_argument('--URI', default='http://localhost:8007', type=str)
args = parser.parse_args()
#
gen_seed = args.gen_seed
inf_seed = args.inf_seed
num_clusters = args.num_clusters
num_cols = args.num_cols
num_rows = args.num_rows
num_splits = args.num_splits
max_mean = args.max_mean
max_std = args.max_std
num_transitions = args.num_transitions
N_GRID = args.N_GRID
URI = args.URI


# create the data
T, M_r, M_c = du.gen_factorial_data_objects(
    gen_seed, num_clusters,
    num_cols, num_rows, num_splits,
    max_mean=max_mean, max_std=max_std,
    )

#
engine = JSONRPCEngine(inf_seed, URI=URI)

# initialize
X_L, X_D = engine.initialize(M_c, M_r, T)

# analyze without do_diagnostics or do_timing
X_L, X_D = engine.analyze(M_c, T, X_L, X_D, n_steps=num_transitions)

# analyze with do_diagnostics
X_L, X_D, diagnostics_dict = engine.analyze(M_c, T, X_L, X_D, n_steps=num_transitions, do_diagnostics=True)

# analyze with do_timing
X_L, X_D, timing_list = engine.analyze(M_c, T, X_L, X_D, n_steps=num_transitions, do_timing=True)

## draw sample states
#for sample_idx in range(num_samples):
#    print "starting sample_idx #: %s" % sample_idx
#    X_L, X_D = engine.analyze(M_c, T, X_L, X_D, kernel_list, lag,
#                              c, r, max_iterations, max_time)
#    p_State = State.p_State(M_c, T, X_L, X_D, N_GRID=N_GRID)
#    plot_filename = 'sample_%s_X_D' % sample_idx
#    pkl_filename = 'sample_%s_pickled_state.pkl.gz' % sample_idx
#    p_State.save(filename=pkl_filename, M_c=M_c, T=T)
#    p_State.plot(filename=plot_filename)
