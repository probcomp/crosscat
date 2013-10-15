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
# Loading a saved model for further analysis
import crosscat.utils.file_utils as f_utils
import crosscat.utils.plot_utils as pu
import crosscat.CrossCatClient as ccc


# 1. Load saved state
pkl_filename = 'flight_data_saved_model.pkl.gz'
load_dict = f_utils.unpickle(filename = pkl_filename)
X_L_list = load_dict['X_L_list']
X_D_list = load_dict['X_D_list']
T = load_dict['T']
M_c = load_dict['M_c']

# 2. Create and visualize column dependency matrix
filebase = 'flight_data_saved'
zplot_filename = '{!s}_feature_z_pre'.format(filebase)
pu.do_gen_feature_z(X_L_list, X_D_list, M_c, zplot_filename)

# 3. Continue transitioning the Markov Chain 
X_L_list_new = []
X_D_list_new = []
num_transitions = 10
numChains = len(X_L_list)
engine = ccc.get_CrossCatClient('local', seed = 0)

for chain_idx in range(numChains):
    print 'Chain {!s}'.format(chain_idx)
    X_L_in = X_L_list[chain_idx]
    X_D_in = X_D_list[chain_idx]
    X_L_prime, X_D_prime = engine.analyze(M_c, T, X_L_in, X_D_in, kernel_list=(),
                                          n_steps=num_transitions)
    X_L_list_new.append(X_L_prime)
    X_D_list_new.append(X_D_prime)

# 4. Create and visualize column dependency matrix again

zplot_filename = '{!s}_feature_z_post'.format(filebase)
pu.do_gen_feature_z(X_L_list_new, X_D_list_new, M_c, zplot_filename)
