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
import os
#
import crosscat.utils.xnet_utils as xu
import crosscat.settings as S
from crosscat.settings import Hadoop as hs


# settings
n_chains = 20
n_steps = 20
#
filename = os.path.join(S.path.web_resources_data_dir, 'dha_small.csv')
script_name = 'hadoop_line_processor.py'
#
table_data_filename = hs.default_table_data_filename
initialize_input_filename = 'initialize_input'
initialize_output_filename = 'initialize_output'
initialize_args_dict = hs.default_initialize_args_dict
analyze_input_filename = 'analyze_input'
analyze_output_filename = 'analyze_output'
analyze_args_dict = hs.default_analyze_args_dict

# set up
table_data = xu.read_and_pickle_table_data(filename, table_data_filename)

# create initialize input
xu.write_initialization_files(initialize_input_filename,
                              initialize_args_dict=initialize_args_dict,
                              n_chains=n_chains)

# initialize
xu.run_script_local(initialize_input_filename, script_name,
                    initialize_output_filename)

# read initialization output, write analyze input
analyze_args_dict['n_steps'] = n_steps
analyze_args_dict['max_time'] = 20
xu.link_initialize_to_analyze(initialize_output_filename,
                              analyze_input_filename,
                              analyze_args_dict)

# analyze
xu.run_script_local(analyze_input_filename, script_name,
                    analyze_output_filename)
