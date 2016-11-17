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

import crosscat.utils.data_utils as du
from crosscat.CrossCatClient import get_CrossCatClient

ccc = get_CrossCatClient('local', seed=0)

gen_seed = 0
num_clusters = 4
num_cols = 8
num_rows = 16
num_splits = 1
max_mean = 10
max_std = 0.1
rng = random.Random(gen_seed)
get_next_seed = lambda: rng.randint(1, 2**31 - 1)
T, M_r, M_c = du.gen_factorial_data_objects(
    get_next_seed(), num_clusters,
    num_cols, num_rows, num_splits,
    max_mean=max_mean, max_std=max_std,
    )

X_L, X_D, = ccc.initialize(M_c, M_r, T, get_next_seed())
X_L_prime, X_D_prime = ccc.analyze(M_c, T, X_L, X_D, get_next_seed())
X_L_prime, X_D_prime = ccc.analyze(M_c, T, X_L_prime, X_D_prime,
    get_next_seed())
