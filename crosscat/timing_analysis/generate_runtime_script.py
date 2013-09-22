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
import itertools


n_steps = 10
#
base_str = ' '.join([
  'python runtime_scripting.py',
  '--num_rows %s',
  '--num_cols %s',
  '--num_clusters %s',
  '--num_splits %s',
  '--n_steps %s' % n_steps,
  '-do_local >>out 2>>err &',
  ])

# num_rows_list = [100, 400, 1000, 4000, 10000]
# num_cols_list = [4, 8, 16, 24, 32]
# num_clusters_list = [10, 20, 30, 40, 50]
# num_splits_list = [1, 2, 3, 4, 5]

num_rows_list = [100, 400]
num_cols_list = [4, 16]
num_clusters_list = [10, 20]
num_splits_list = [1, 2]

take_product_of = [num_rows_list, num_cols_list, num_clusters_list, num_splits_list]
for num_rows, num_cols, num_clusters, num_splits \
    in itertools.product(*take_product_of):
  this_base_str = base_str % (num_rows, num_cols, num_clusters, num_splits)
  print this_base_str
