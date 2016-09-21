#
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
#

from __future__ import print_function

import multiprocessing

import crosscat.LocalEngine as LE
import crosscat.utils.sample_utils as su


class MultiprocessingEngine(LE.LocalEngine):
    """A simple interface to the Cython-wrapped C++ engine.

    MultiprocessingEngine holds no state.
    Methods use resources on the local machine.
    """

    def __init__(self, seed=None, cpu_count=None):
        super(MultiprocessingEngine, self).__init__(seed=None)
        self.pool = multiprocessing.Pool(cpu_count)
        self.mapper = self.pool.map
        return

    def __enter__(self):
        return self

    def __del__(self):
        self.pool.terminate()

    def __exit__(self, type, value, traceback):
        self.pool.terminate()
