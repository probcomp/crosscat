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

import random

import crosscat.utils.convergence_test_utils as ctu
import crosscat.utils.data_utils as du

from crosscat.MultiprocessingEngine import MultiprocessingEngine

# settings
gen_seed = 0
inf_seed = 0
num_clusters = 4
num_cols = 32
num_rows = 400
num_views = 2
n_steps = 1
n_times = 5
n_chains = 3
n_test = 100
rng = random.Random(gen_seed)
get_next_seed = lambda: rng.randint(1, 2**31 - 1)

# generate some data
T, M_r, M_c, data_inverse_permutation_indices = du.gen_factorial_data_objects(
        get_next_seed(), num_clusters, num_cols, num_rows, num_views,
        max_mean=100, max_std=1, send_data_inverse_permutation_indices=True)

view_assignment_truth, X_D_truth = ctu.truth_from_permute_indices(
        data_inverse_permutation_indices, num_rows, num_cols,
        num_views, num_clusters)

X_L_gen, X_D_gen = du.get_generative_clustering(get_next_seed(),
        M_c, M_r, T, data_inverse_permutation_indices, num_clusters, num_views)
T_test = ctu.create_test_set(M_c, T, X_L_gen, X_D_gen, n_test, seed_seed=0)

#
generative_mean_test_log_likelihood = ctu.calc_mean_test_log_likelihood(M_c, T,
        X_L_gen, X_D_gen, T_test)

# run some tests
engine = MultiprocessingEngine()

# single state test
single_state_ARIs = []
single_state_mean_test_lls = []
X_L, X_D = engine.initialize(M_c, M_r, T, get_next_seed(), n_chains=1)
single_state_ARIs.append(ctu.get_column_ARI(X_L, view_assignment_truth))
single_state_mean_test_lls.append(
    ctu.calc_mean_test_log_likelihood(M_c, T, X_L, X_D, T_test))

for time_i in range(n_times):
    X_L, X_D = engine.analyze(M_c, T, X_L, X_D, get_next_seed(),
        n_steps=n_steps)
    single_state_ARIs.append(ctu.get_column_ARI(X_L, view_assignment_truth))
    single_state_mean_test_lls.append(
        ctu.calc_mean_test_log_likelihood(M_c, T, X_L, X_D, T_test))

# multistate test
multi_state_ARIs = []
multi_state_mean_test_lls = []
X_L_list, X_D_list = engine.initialize(
    M_c, M_r, T, get_next_seed(), n_chains=n_chains)
