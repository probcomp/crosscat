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
import itertools
import collections
import copy
#
import numpy
#
import crosscat.cython_code.State as State
import crosscat.EngineTemplate as EngineTemplate
import crosscat.utils.sample_utils as su
import crosscat.utils.general_utils as gu
import crosscat.utils.inference_utils as iu
# for default_diagnostic_func_dict below
import crosscat.utils.diagnostic_utils

class LocalEngine(EngineTemplate.EngineTemplate):

    """A simple interface to the Cython-wrapped C++ engine

    LocalEngine holds no state other than a seed generator.
    Methods use resources on the local machine.

    """

    def __init__(self, seed=None):
        """Initialize a LocalEngine

        This is really just setting the initial seed to be used for
        initializing CrossCat states.  Seeds are generated sequentially

        """
        super(LocalEngine, self).__init__(seed=seed)
        self.mapper = map
        self.do_initialize = _do_initialize_tuple
        self.do_analyze = _do_analyze_tuple
        self.do_insert = _do_insert_tuple
        return

    def get_initialize_arg_tuples(self, M_c, M_r, T, initialization,
                                  row_initialization, n_chains,
                                  ROW_CRP_ALPHA_GRID,
                                  COLUMN_CRP_ALPHA_GRID,
                                  S_GRID, MU_GRID,
                                  N_GRID,
                                  ):
        seeds = [self.get_next_seed() for seed_idx in range(n_chains)]
        arg_tuples = itertools.izip(
            seeds,
            itertools.cycle([M_c]),
            itertools.cycle([M_r]),
            itertools.cycle([T]),
            itertools.cycle([initialization]),
            itertools.cycle([row_initialization]),
            itertools.cycle([ROW_CRP_ALPHA_GRID]),
            itertools.cycle([COLUMN_CRP_ALPHA_GRID]),
            itertools.cycle([S_GRID]),
            itertools.cycle([MU_GRID]),
            itertools.cycle([N_GRID]),
        )
        return arg_tuples

    def initialize(self, M_c, M_r, T, initialization='from_the_prior',
                   row_initialization=-1, n_chains=1,
                   ROW_CRP_ALPHA_GRID=(),
                   COLUMN_CRP_ALPHA_GRID=(),
                   S_GRID=(), MU_GRID=(),
                   N_GRID=31,
                   # subsample=False,
                   # subsample_proportion=None,
                   # subsample_rows_list=None,
                   ):
        """Sample a latent state from prior

        :param M_c: The column metadata
        :type M_c: dict
        :param M_r: The row metadata
        :type M_r: dict
        :param T: The data table in mapped representation (all floats, generated
                  by data_utils.read_data_objects)
        :type T: list of lists
        :returns: X_L, X_D -- the latent state

        """

        # FIXME: why is M_r passed?
        arg_tuples = self.get_initialize_arg_tuples(
            M_c, M_r, T, initialization,
            row_initialization, n_chains,
            ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
            S_GRID, MU_GRID,
            N_GRID,
        )
        chain_tuples = self.mapper(self.do_initialize, arg_tuples)

        X_L_list, X_D_list = zip(*chain_tuples)
        if n_chains == 1:
            X_L_list, X_D_list = X_L_list[0], X_D_list[0]
        return X_L_list, X_D_list


    def get_insert_arg_tuples(self, M_c, T, X_L_list, X_D_list, new_rows, N_GRID, CT_KERNEL):
        n_chains = len(X_L_list)
        # seeds = [self.get_next_seed() for seed_idx in range(n_chains)]
        arg_tuples = itertools.izip(
            itertools.cycle([M_c]),
            itertools.cycle([T]),
            X_L_list, X_D_list,
            itertools.cycle([new_rows]),
            itertools.cycle([N_GRID]),
            itertools.cycle([CT_KERNEL]),
        )
        return arg_tuples


    def insert(self, M_c, T, X_L_list, X_D_list, new_rows=None, N_GRID=31, CT_KERNEL=0):
        """
        Insert mutates the data T. 
        """

        if new_rows is None:
            raise ValueError("new_row must exist")

        if not isinstance(new_rows, list):
            raise TypeError('new_rows must be list of lists')
            if not isinstance(new_rows[0], list):
                raise TypeError('new_rows must be list of lists')

        X_L_list, X_D_list, was_multistate = su.ensure_multistate(X_L_list, X_D_list)

        # get insert arg tuples
        arg_tuples = self.get_insert_arg_tuples(M_c, T, X_L_list, X_D_list, new_rows, N_GRID, CT_KERNEL)

        chain_tuples = self.mapper(self.do_insert, arg_tuples)
        X_L_list, X_D_list = zip(*chain_tuples)

        if not was_multistate:
            X_L_list, X_D_list = X_L_list[0], X_D_list[0]

        T.extend(new_rows)

        ret_tuple = X_L_list, X_D_list, T

        return ret_tuple


    def get_analyze_arg_tuples(self, M_c, T, X_L_list, X_D_list, kernel_list,
                               n_steps, c, r, max_iterations, max_time, diagnostic_func_dict, every_N,
                               ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
                               S_GRID, MU_GRID,
                               N_GRID,
                               do_timing,
                               CT_KERNEL,
                               ):
        n_chains = len(X_L_list)
        seeds = [self.get_next_seed() for seed_idx in range(n_chains)]
        arg_tuples = itertools.izip(
            seeds,
            X_L_list, X_D_list,
            itertools.cycle([M_c]),
            itertools.cycle([T]),
            itertools.cycle([kernel_list]),
            itertools.cycle([n_steps]),
            itertools.cycle([c]),
            itertools.cycle([r]),
            itertools.cycle([max_iterations]),
            itertools.cycle([max_time]),
            itertools.cycle([diagnostic_func_dict]),
            itertools.cycle([every_N]),
            itertools.cycle([ROW_CRP_ALPHA_GRID]),
            itertools.cycle([COLUMN_CRP_ALPHA_GRID]),
            itertools.cycle([S_GRID]),
            itertools.cycle([MU_GRID]),
            itertools.cycle([N_GRID]),
            itertools.cycle([do_timing]),
            itertools.cycle([CT_KERNEL]),
        )
        return arg_tuples

    def analyze(self, M_c, T, X_L, X_D, kernel_list=(), n_steps=1, c=(), r=(),
                max_iterations=-1, max_time=-1, do_diagnostics=False,
                diagnostics_every_N=1,
                ROW_CRP_ALPHA_GRID=(),
                COLUMN_CRP_ALPHA_GRID=(),
                S_GRID=(), MU_GRID=(),
                N_GRID=31,
                do_timing=False,
                CT_KERNEL=0,
                ):
        """Evolve the latent state by running MCMC transition kernels

        :param M_c: The column metadata
        :type M_c: dict
        :param T: The data table in mapped representation (all floats, generated
                  by data_utils.read_data_objects)
        :param X_L: the latent variables associated with the latent state
        :type X_L: dict
        :param X_D: the particular cluster assignments of each row in each view
        :type X_D: list of lists
        :param kernel_list: names of the MCMC transition kernels to run
        :type kernel_list: list of strings
        :param n_steps: the number of times to run each MCMC transition kernel
        :type n_steps: int
        :param c: the (global) column indices to run MCMC transition kernels on
        :type c: list of ints
        :param r: the (global) row indices to run MCMC transition kernels on
        :type r: list of ints
        :param max_iterations: the maximum number of times ot run each MCMC
                               transition kernel. Applicable only if
                               max_time != -1.
        :type max_iterations: int
        :param max_time: the maximum amount of time (seconds) to run MCMC
                         transition kernels for before stopping to return
                         progress
        :type max_time: float
        :returns: X_L, X_D -- the evolved latent state

        """
        if n_steps <= 0:
            raise ValueError("You must do at least one analyze step.")

        if CT_KERNEL not in [0,1]:
            raise ValueError("CT_KERNEL must be 0 (Gibbs) or 1 (MH)")

        if do_timing:
            # diagnostics and timing are exclusive
            do_diagnostics = False
        diagnostic_func_dict, reprocess_diagnostics_func = do_diagnostics_to_func_dict(
            do_diagnostics)
        X_L_list, X_D_list, was_multistate = su.ensure_multistate(X_L, X_D)
        arg_tuples = self.get_analyze_arg_tuples(M_c, T, X_L_list, X_D_list,
                                                 kernel_list, n_steps, c, r, max_iterations, max_time,
                                                 diagnostic_func_dict, diagnostics_every_N,
                                                 ROW_CRP_ALPHA_GRID,
                                                 COLUMN_CRP_ALPHA_GRID,
                                                 S_GRID, MU_GRID,
                                                 N_GRID,
                                                 do_timing,
                                                 CT_KERNEL,
                                                 )
        chain_tuples = self.mapper(self.do_analyze, arg_tuples)
        X_L_list, X_D_list, diagnostics_dict_list = zip(*chain_tuples)
        if do_timing:
            timing_list = diagnostics_dict_list
        if not was_multistate:
            X_L_list, X_D_list = X_L_list[0], X_D_list[0]
        ret_tuple = X_L_list, X_D_list
        #
        if diagnostic_func_dict is not None:
            diagnostics_dict = munge_diagnostics(diagnostics_dict_list)
            if reprocess_diagnostics_func is not None:
                diagnostics_dict = reprocess_diagnostics_func(diagnostics_dict)
            ret_tuple = ret_tuple + (diagnostics_dict, )
        if do_timing:
            ret_tuple = ret_tuple + (timing_list, )
        return ret_tuple

    def _sample_and_insert(self, M_c, T, X_L, X_D, matching_row_indices):
        p_State = State.p_State(M_c, T, X_L, X_D)
        draws = []
        for matching_row_idx in matching_row_indices:
            random_seed = self.get_next_seed()
            draw = p_State.get_draw(matching_row_idx, random_seed)
            p_State.insert_row(draw, matching_row_idx)
            draws.append(draw)
            T.append(draw)
        X_L, X_D = p_State.get_X_L(), p_State.get_X_D()
        return draws, T, X_L, X_D

    def sample_and_insert(self, M_c, T, X_L, X_D, matching_row_idx):
        matching_row_indices = gu.ensure_listlike(matching_row_idx)
        if len(matching_row_indices) == 0:
            matching_row_indices = range(len(T))
            pass
        was_single_row = len(matching_row_indices) == 1
        draws, T, X_L, X_D = self._sample_and_insert(M_c, T, X_L, X_D,
                matching_row_indices)
        if was_single_row:
            draws = draws[0]
            pass
        return draws, T, X_L, X_D

    def simple_predictive_sample(self, M_c, X_L, X_D, Y, Q, n=1):
        """Sample values from the predictive distribution of the given latent state

        :param M_c: The column metadata
        :type M_c: dict
        :param X_L: the latent variables associated with the latent state
        :type X_L: dict
        :param X_D: the particular cluster assignments of each row in each view
        :type X_D: list of lists
        :param Y: A list of constraints to apply when sampling.  Each constraint
                  is a triplet of (r, d, v): r is the row index, d is the column
                  index and v is the value of the constraint
        :type Y: list of lists
        :param Q: A list of values to sample.  Each value is doublet of (r, d):
                  r is the row index, d is the column index
        :type Q: list of lists
        :param n: the number of samples to draw
        :type n: int
        :returns: list of floats -- samples in the same order specified by Q

        """
        get_next_seed = self.get_next_seed
        samples = _do_simple_predictive_sample(
            M_c, X_L, X_D, Y, Q, n, get_next_seed)
        return samples

    def simple_predictive_probability(self, M_c, X_L, X_D, Y, Q):
        """Calculate the probability of a cell taking a value given a latent state

        :param M_c: The column metadata
        :type M_c: dict
        :param X_L: the latent variables associated with the latent state
        :type X_L: dict
        :param X_D: the particular cluster assignments of each row in each view
        :type X_D: list of lists
        :param Y: A list of constraints to apply when sampling.  Each constraint
                  is a triplet of (r, d, v): r is the row index, d is the column
                  index and v is the value of the constraint
        :type Y: list of lists
        :param Q: A list of values to sample.  Each value is doublet of (r, d):
                  r is the row index, d is the column index
        :type Q: list of lists
        :returns: list of floats -- probabilities of the values specified by Q

        """
        return su.simple_predictive_probability(M_c, X_L, X_D, Y, Q)

    def simple_predictive_probability_multistate(self, M_c, X_L_list, X_D_list, Y, Q):
        """Calculate the probability of a cell taking a value given a latent state

        :param M_c: The column metadata
        :type M_c: dict
        :param X_L_list: list of the latent variables associated with the latent state
        :type X_L_list: list of dict
        :param X_D_list: list of the particular cluster assignments of each row in each view
        :type X_D_list: list of list of lists
        :param Y: A list of constraints to apply when sampling.  Each constraint
                  is a triplet of (r, d, v): r is the row index, d is the column
                  index and v is the value of the constraint
        :type Y: list of lists
        :param Q: A list of values to sample.  Each value is doublet of (r, d):
                  r is the row index, d is the column index
        :type Q: list of lists
        :returns: list of floats -- probabilities of the values specified by Q

        """
        return su.simple_predictive_probability_multistate(M_c, X_L_list, X_D_list, Y, Q)

    def mutual_information(self, M_c, X_L_list, X_D_list, Q, n_samples=1000):
        """
        Return the estimated mutual information for each pair of columns on Q given
        the set of samples.
        
        :param M_c: The column metadata
        :type M_c: dict
        :param X_L_list: list of the latent variables associated with the latent state
        :type X_L_list: list of dict
        :param X_D_list: list of the particular cluster assignments of each row in each view
        :type X_D_list: list of list of lists
        :param Q: List of tuples where each tuple contains the two column indexes to compare
        :type Q: list of two-tuples of ints
        :param n_samples: the number of simple predictive samples to use
        :type n_samples: int
        :returns: list of list, where each sublist is a set of MIs and Linfoots from each crosscat sample.
        """
        return iu.mutual_information(M_c, X_L_list, X_D_list, Q, n_samples)

    def row_structural_typicality(self, X_L_list, X_D_list, row_id):
        """
        Returns the typicality (opposite of anomalousness) of the given row.
        
        :param X_L_list: list of the latent variables associated with the latent state
        :type X_L_list: list of dict
        :param X_D_list: list of the particular cluster assignments of each row in each view
        :type X_D_list: list of list of lists
        :param row_id: id of the target row
        :type row_id: int
        :returns: float, the typicality, from 0 to 1
        """
        return su.row_structural_typicality(X_L_list, X_D_list, row_id)

    def column_structural_typicality(self, X_L_list, col_id):
        """
        Returns the typicality (opposite of anomalousness) of the given column.
        
        :param X_L_list: list of the latent variables associated with the latent state
        :type X_L_list: list of dict
        :param col_id: id of the target col
        :type col_id: int
        :returns: float, the typicality, from 0 to 1
        """
        return su.column_structural_typicality(X_L_list, col_id)

    def similarity(self, M_c, X_L_list, X_D_list, given_row_id, target_row_id, target_columns=None):
        """Computes the similarity of the given row to the target row, averaged over all the
        column indexes given by target_columns.

        :param M_c: The column metadata
        :type M_c: dict
        :param X_L: list of the latent variables associated with the latent state
        :type X_L: list of dicts
        :param X_D: list of the particular cluster assignments of each row in each view
        :type X_D: list of list of lists
        :param given_row_id: the id of one of the rows to measure similarity between
        :type given_row_id: int
        :param target_row_id: the id of the other row to measure similarity between
        :type target_row_id: int
        :param target_columns: the columns to average the similarity over. defaults to all columns.
        :type target_columns: int, string, or list of ints
        :returns: float

        """
        return su.similarity(M_c, X_L_list, X_D_list, given_row_id, target_row_id, target_columns)

    def impute(self, M_c, X_L, X_D, Y, Q, n):
        """Impute values from the predictive distribution of the given latent state

        :param M_c: The column metadata
        :type M_c: dict
        :param X_L: the latent variables associated with the latent state
        :type X_L: dict
        :param X_D: the particular cluster assignments of each row in each view
        :type X_D: list of lists
        :param Y: A list of constraints to apply when sampling.  Each constraint
                  is a triplet of (r, d, v): r is the row index, d is the column
                  index and v is the value of the constraint
        :type Y: list of lists
        :param Q: A list of values to sample.  Each value is doublet of (r, d):
                  r is the row index, d is the column index
        :type Q: list of lists
        :param n: the number of samples to use in the imputation
        :type n: int
        :returns: list of floats -- imputed values in the same order as
                  specified by Q

        """
        e = su.impute(M_c, X_L, X_D, Y, Q, n, self.get_next_seed)
        return e

    def impute_and_confidence(self, M_c, X_L, X_D, Y, Q, n):
        """Impute values and confidence of the value from the predictive
        distribution of the given latent state

        :param M_c: The column metadata
        :type M_c: dict
        :param X_L: the latent variables associated with the latent state
        :type X_L: dict
        :param X_D: the particular cluster assignments of each row in each view
        :type X_D: list of lists
        :param Y: A list of constraints to apply when sampling.  Each constraint
                  is a triplet of (r, d, v): r is the row index, d is the column
                  index and v is the value of the constraint
        :type Y: list of lists
        :param Q: A list of values to sample.  Each value is doublet of (r, d):
                  r is the row index, d is the column index
        :type Q: list of lists
        :param n: the number of samples to use in the imputation
        :type n: int
        :returns: list of lists -- list of (value, confidence) tuples in the
                  same order as specified by Q

        """
        if isinstance(X_L, (list, tuple)):
            assert isinstance(X_D, (list, tuple))
            # TODO: multistate impute doesn't exist yet
            #e,confidence = su.impute_and_confidence_multistate(M_c, X_L, X_D, Y, Q, n, self.get_next_seed)
            e, confidence = su.impute_and_confidence(
                M_c, X_L, X_D, Y, Q, n, self.get_next_seed)
        else:
            e, confidence = su.impute_and_confidence(
                M_c, X_L, X_D, Y, Q, n, self.get_next_seed)
        return (e, confidence)


def do_diagnostics_to_func_dict(do_diagnostics):
    diagnostic_func_dict = None
    reprocess_diagnostics_func = None
    if do_diagnostics:
        if isinstance(do_diagnostics, (dict,)):
            diagnostic_func_dict = do_diagnostics
        else:
            diagnostic_func_dict = dict(default_diagnostic_func_dict)
        if 'reprocess_diagnostics_func' in diagnostic_func_dict:
            reprocess_diagnostics_func = diagnostic_func_dict.pop(
                'reprocess_diagnostics_func')
    return diagnostic_func_dict, reprocess_diagnostics_func


def get_value_in_each_dict(key, dict_list):
    return numpy.array([dict_i[key] for dict_i in dict_list]).T


def munge_diagnostics(diagnostics_dict_list):
    # all dicts should have the same keys
    diagnostic_names = diagnostics_dict_list[0].keys()
    diagnostics_dict = {
        diagnostic_name: get_value_in_each_dict(diagnostic_name, diagnostics_dict_list)
        for diagnostic_name in diagnostic_names
    }
    return diagnostics_dict

# switched ordering so args that change come first
# FIXME: change LocalEngine.initialze to match ordering here


def _do_initialize(SEED, M_c, M_r, T, initialization, row_initialization,
                   ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
                   S_GRID, MU_GRID,
                   N_GRID,
                   ):
    p_State = State.p_State(M_c, T, initialization=initialization,
                            row_initialization=row_initialization, SEED=SEED,
                            ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
                            COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
                            S_GRID=S_GRID,
                            MU_GRID=MU_GRID,
                            N_GRID=N_GRID,
                            )
    X_L = p_State.get_X_L()
    X_D = p_State.get_X_D()
    return X_L, X_D


def _do_initialize_tuple(arg_tuple):
    return _do_initialize(*arg_tuple)


def _do_insert_tuple(arg_tuple):
    return _do_insert(*arg_tuple)

def _do_insert( M_c, T, X_L, X_D, new_rows, N_GRID, CT_KERNEL):

    p_State = State.p_State(M_c, T, X_L=X_L, X_D=X_D,
                            N_GRID=N_GRID,
                            CT_KERNEL=CT_KERNEL)

    row_idx = len(T)
    for row_data in new_rows:
        p_State.insert_row(row_data, row_idx)
        p_State.transition(which_transitions=['row_partition_assignments'], r=[row_idx])
        row_idx += 1

    X_L_prime = p_State.get_X_L()
    X_D_prime = p_State.get_X_D()
    return X_L_prime, X_D_prime

# switched ordering so args that change come first
# FIXME: change LocalEngine.analyze to match ordering here

def _do_analyze(SEED, X_L, X_D, M_c, T, kernel_list, n_steps, c, r,
                max_iterations, max_time,
                ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
                S_GRID, MU_GRID,
                N_GRID,
                CT_KERNEL,
                ):
    p_State = State.p_State(M_c, T, X_L, X_D, SEED=SEED,
                            ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
                            COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
                            S_GRID=S_GRID,
                            MU_GRID=MU_GRID,
                            N_GRID=N_GRID,
                            CT_KERNEL=CT_KERNEL
                            )
    p_State.transition(kernel_list, n_steps, c, r,
                       max_iterations, max_time)
    X_L_prime = p_State.get_X_L()
    X_D_prime = p_State.get_X_D()
    return X_L_prime, X_D_prime


def _do_analyze_tuple(arg_tuple):
    return _do_analyze_with_diagnostic(*arg_tuple)

def get_child_n_steps_list(n_steps, every_N):
    if every_N is None:
        # results in one block of size n_steps
        every_N = n_steps
    missing_endpoint = numpy.arange(0, n_steps, every_N)
    with_endpoint = numpy.append(missing_endpoint, n_steps)
    child_n_steps_list = numpy.diff(with_endpoint)
    return child_n_steps_list.tolist()

none_summary = lambda p_State: None

# switched ordering so args that change come first
# FIXME: change LocalEngine.analyze to match ordering here


def _do_analyze_with_diagnostic(SEED, X_L, X_D, M_c, T, kernel_list, n_steps, c, r,
        max_iterations, max_time, diagnostic_func_dict, every_N,
        ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
        S_GRID, MU_GRID,
        N_GRID,
        do_timing,
        CT_KERNEL,
        ):
    diagnostics_dict = collections.defaultdict(list)
    if diagnostic_func_dict is None:
        diagnostic_func_dict = dict()
        every_N = None
    child_n_steps_list = get_child_n_steps_list(n_steps, every_N)
    #
    p_State = State.p_State(M_c, T, X_L, X_D, SEED=SEED,
                            ROW_CRP_ALPHA_GRID=ROW_CRP_ALPHA_GRID,
                            COLUMN_CRP_ALPHA_GRID=COLUMN_CRP_ALPHA_GRID,
                            S_GRID=S_GRID,
                            MU_GRID=MU_GRID,
                            N_GRID=N_GRID,
                            CT_KERNEL=CT_KERNEL,
                            )
    with gu.Timer('all transitions', verbose=False) as timer:
        for child_n_steps in child_n_steps_list:
            p_State.transition(kernel_list, child_n_steps, c, r,
                               max_iterations, max_time)
            for diagnostic_name, diagnostic_func in diagnostic_func_dict.iteritems():
                diagnostic_value = diagnostic_func(p_State)
                diagnostics_dict[diagnostic_name].append(diagnostic_value)
                pass
            pass
        pass
    X_L_prime = p_State.get_X_L()
    X_D_prime = p_State.get_X_D()
    #
    if do_timing:
        # diagnostics and timing are exclusive
        diagnostics_dict = timer.elapsed_secs
        pass
    return X_L_prime, X_D_prime, diagnostics_dict


def _do_simple_predictive_sample(M_c, X_L, X_D, Y, Q, n, get_next_seed):
    is_multistate = su.get_is_multistate(X_L, X_D)
    if is_multistate:
        samples = su.simple_predictive_sample_multistate(M_c, X_L, X_D, Y, Q,
                                                         get_next_seed, n)
    else:
        samples = su.simple_predictive_sample(M_c, X_L, X_D, Y, Q,
                                              get_next_seed, n)
    return samples


default_diagnostic_func_dict = dict(
    # fully qualify path b/c dview.sync_imports can't deal with 'as'
    # imports
    logscore=crosscat.utils.diagnostic_utils.get_logscore,
    num_views=crosscat.utils.diagnostic_utils.get_num_views,
    column_crp_alpha=crosscat.utils.diagnostic_utils.get_column_crp_alpha,
    # any outputs required by reproess_diagnostics_func must be generated
    # as well
    column_partition_assignments=crosscat.utils.diagnostic_utils.get_column_partition_assignments,
    reprocess_diagnostics_func=crosscat.utils.diagnostic_utils.default_reprocess_diagnostics_func,
)


if __name__ == '__main__':
    import crosscat.utils.data_utils as du
    import crosscat.utils.convergence_test_utils as ctu
    import crosscat.utils.timing_test_utils as ttu

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
    CT_KERNEL = 1

    # generate some data
    T, M_r, M_c, data_inverse_permutation_indices = du.gen_factorial_data_objects(
        gen_seed, num_clusters, num_cols, num_rows, num_views,
        max_mean=100, max_std=1, send_data_inverse_permutation_indices=True)
    view_assignment_truth, X_D_truth = ctu.truth_from_permute_indices(
        data_inverse_permutation_indices, num_rows, num_cols, num_views, num_clusters)

    # There is currently (4/7/2014) no ttu.get_generative_clustering function which 
    # X_L_gen, X_D_gen = ttu.get_generative_clustering(M_c, M_r, T,
    #                                                  data_inverse_permutation_indices, num_clusters, num_views)
    # T_test = ctu.create_test_set(M_c, T, X_L_gen, X_D_gen, n_test, seed_seed=0)
    # #
    # generative_mean_test_log_likelihood = ctu.calc_mean_test_log_likelihood(
    #     M_c, T,
    #     X_L_gen, X_D_gen, T_test)

    # run some tests
    engine = LocalEngine(seed=inf_seed)
    multi_state_ARIs = []
    multi_state_mean_test_lls = []
    X_L_list, X_D_list = engine.initialize(M_c, M_r, T, n_chains=n_chains)
    multi_state_ARIs.append(
        ctu.get_column_ARIs(X_L_list, view_assignment_truth))

    # multi_state_mean_test_lls.append(ctu.calc_mean_test_log_likelihoods(M_c, T,
    #                                                                     X_L_list, X_D_list, T_test))
    for time_i in range(n_times):
        X_L_list, X_D_list = engine.analyze(
            M_c, T, X_L_list, X_D_list, n_steps=n_steps, CT_KERNEL=CT_KERNEL)
        multi_state_ARIs.append(
            ctu.get_column_ARIs(X_L_list, view_assignment_truth))
        # multi_state_mean_test_lls.append(
        #     ctu.calc_mean_test_log_likelihoods(M_c, T,
        #                                        X_L_list, X_D_list, T_test))

    X_L_list, X_D_list, diagnostics_dict = engine.analyze(
        M_c, T, X_L_list, X_D_list,
        n_steps=n_steps, do_diagnostics=True)

    # print results
    ct_kernel_name = 'UNKNOWN'
    if CT_KERNEL==0:
        ct_kernel_name = 'GIBBS'
    elif CT_KERNEL==1:
        ct_kernel_name = 'METROPOLIS'
    
    print 'Running with %s CT_KERNEL' % (ct_kernel_name)
    print 'generative_mean_test_log_likelihood'
    # print generative_mean_test_log_likelihood
    #
    print 'multi_state_mean_test_lls:'
    print multi_state_mean_test_lls
    #
    print 'multi_state_ARIs:'
    print multi_state_ARIs
