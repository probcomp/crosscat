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
# Run the build process by running the command 'python setup.py build'
#
# If everything works well you should find a subdirectory in the build
# subdirectory that contains the files needed to run the application

# Run the build process by running the command 'python setup.py build'
#
# If everything works well you should find a subdirectory in the build
# subdirectory that contains the files needed to run the application


import sys
#
import cx_Freeze


excludes = [
    'FixTk',
    'Tkconstants',
    'Tkinter',
    ]
includes = [
    'crosscat.utils.data_utils',
    'crosscat.utils.file_utils',
    'crosscat.utils.inference_utils',
    'crosscat.utils.mutual_information_test_utils',
    'crosscat.utils.timing_test_utils',
    'crosscat.utils.convergence_test_utils',
    'crosscat.LocalEngine',
    'crosscat.HadoopEngine',
    'crosscat.cython_code.State',
    'crosscat.utils.xnet_utils',
    'crosscat.utils.general_utils',
    'crosscat.utils.sample_utils',
    'numpy',
    'sklearn.metrics',
    'sklearn.utils.lgamma',
    'scipy.special',
    'scipy.sparse.csgraph._validation',
    ]

buildOptions = dict(
        excludes = excludes,
        includes = includes,
        compressed = False,
)

executables = [
        cx_Freeze.Executable("hadoop_line_processor.py", base = None)
]

cx_Freeze.setup(
        name = "hadoop_line_processor",
        version = "0.1",
        description = "process arbitrary engine commands on hadoop",
        executables = executables,
        options = dict(build_exe = buildOptions))
