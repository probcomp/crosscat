#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka, Avinash Gandhe
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
# Creating a model and saving the state

# 1. Import packages/modules needed
import crosscat.utils.file_utils as f_utils
import crosscat.utils.data_utils as du
import crosscat.CrossCatClient as ccc

# 2. Load a data table from csv file. In this example, we use synthetic data
filename = 'flight_data_subset_missing_data.csv'
T, M_r, M_c = du.read_model_data_from_csv(filename, gen_seed=0)

# 3. Initialize CrossCat Engine and Build Model
engine = ccc.get_CrossCatClient('local', seed = 0)
X_L_list = []
X_D_list = []
numChains = 10
num_transitions = 10

for chain_idx in range(numChains):
    print 'Chain {!s}'.format(chain_idx)
    X_L, X_D = engine.initialize(M_c, M_r, T)
    X_L_prime, X_D_prime = engine.analyze(M_c, T, X_L, X_D, kernel_list=(),
                                          n_steps=num_transitions)
    X_L_list.append(X_L_prime)
    X_D_list.append(X_D_prime)

saved_dict = {'T':T, 'M_c':M_c, 'X_L_list':X_L_list, 'X_D_list': X_D_list}
pkl_filename = 'flight_data_saved_model.pkl.gz'
f_utils.pickle(saved_dict, filename = pkl_filename)
