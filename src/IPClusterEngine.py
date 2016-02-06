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
import functools
#
from IPython.parallel import Client
#
import crosscat
import crosscat.LocalEngine as LE


def partialize(func, args_dict, dview):
    # why is this push necessary?
    dview.push(args_dict, block=True)
    helper = functools.partial(func, **args_dict)
    return helper


class IPClusterEngine(LE.LocalEngine):
    """A simple interface to the Cython-wrapped C++ engine

    IPClusterEngine

    """

    def __init__(self, config_filename=None, profile=None, seed=None, sshkey=None, packer='json'):
        """Initialize a IPClusterEngine

        Do IPython.parallel operations to set up cluster and generate mapper.

        """
        super(IPClusterEngine, self).__init__(seed=seed)
        rc = Client(config_filename, profile=profile, sshkey=sshkey, packer=packer)
        # FIXME: add a warning if environment in direct view is not 'empty'?
        #        else, might become dependent on an object created in
        #        environemnt in a prior run
        dview = rc.direct_view()
        lview = rc.load_balanced_view()
        with dview.sync_imports(local=True):
            import crosscat
        mapper = lambda f, tuples: self.lview.map(f, *tuples)
        # if you're trying to debug issues, consider clearning to start fresh
        # rc.clear(block=True)
        #
        self.rc = rc
        self.dview = dview
        self.lview = lview
        self.mapper = mapper
        self.do_initialize = None
        self.do_analyze = None
        return

    def get_initialize_arg_tuples(self, M_c, M_r, T, initialization,
            row_initialization, n_chains):
        args_dict = dict(M_c=M_c, M_r=M_r, T=T, initialization=initialization,
                row_initialization=row_initialization)
        do_initialize = partialize(crosscat.LocalEngine._do_initialize,
                args_dict, self.dview)
        seeds = [self.get_next_seed() for seed_idx in range(n_chains)]
        arg_tuples = [seeds]
        #
        self.do_initialize = do_initialize
        return arg_tuples

    def get_analyze_arg_tuples(self, M_c, T, X_L, X_D, kernel_list=(), n_steps=1, c=(), r=(),
                max_iterations=-1, max_time=-1, diagnostic_func_dict=None, every_N=1):
        n_chains = len(X_L)
        args_dict = dict(M_c=M_c, T=T, kernel_list=kernel_list, n_steps=n_steps,
                c=c, r=r, max_iterations=max_iterations, max_time=max_time,
                diagnostic_func_dict=diagnostic_func_dict, every_N=every_N)
        do_analyze = partialize(crosscat.LocalEngine._do_analyze_with_diagnostic,
                args_dict, self.dview)
        seeds = [self.get_next_seed() for seed_idx in range(n_chains)]
        arg_tuples = [seeds, X_L, X_D]
        #
        self.do_analyze = do_analyze
        return arg_tuples
