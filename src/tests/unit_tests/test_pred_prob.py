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

import pytest

import crosscat.tests.synthetic_data_generator as sdg
from crosscat.LocalEngine import LocalEngine

'''
This test suite ensures that invoking simple_predictive_probability_observed
and simple_predictive_probability_unobserved from sample_utils.py with various
constraint Y and query Q patterns does not throw runtime errors.

TODO:
    Whether simple_predictive_probability performs the 'correct' computation in
    carefully constructed cases is not addressed by this test. The design
    choices for what it means to condition on observed/unobserved members is not
    documented anywhere in the repository and must be reverse engineered from
    the spaghetti code.

    Unfortunatley, there is no one place to figure out how the constraints Y
    are being used in the evaluation of the predictive probability. The most
    important functions which indicate how conditional constraints are used are:

        sample_utils.py
            - def get_draw_constraints(X_L, X_D, Y, draw_row, draw_column)
            - def def get_cluster_sampling_constraints(Y, query_row)

        ComponentModel.h
            - get_draw_constrained(int random_seed,
                    const std::vector<double>& constraints) const;
            - calc_element_predictive_logp_constrained(double element,
                const std::vector<double>& constraints) const;

        It is not entirely clear
            - What is the probabilistic reasoning behind the code that takes
                [Y, Q] and produces constraint_dict?
            - How are the constraints marshalled between Python and CPP?
            - When conditioning on a cell in observed row, do we delete the
                already observed value?
            - Why can we not condition one unobserved on another unobserved?
                This behavior throws a RuntimeError (tested below).
'''

N_ROWS = 1000

def quick_le(seed, n_chains=1):
    # Specify synthetic dataset structure.
    cctypes = ['continuous', 'continuous', 'multinomial', 'multinomial',
        'continuous']
    distargs = [None, None, dict(K=9), dict(K=7), None]
    cols_to_views = [0, 0, 0, 1, 1]
    separation = [0.6, 0.9]
    cluster_weights = [[.2, .3, .5],[.9, .1]]

    # Obtain the generated dataset and metadata/
    T, M_c, M_r = sdg.gen_data(cctypes, N_ROWS, cols_to_views, cluster_weights,
        separation, seed=seed, distargs=distargs, return_structure=True)

    # Create, initialize, and analyze the engine.
    engine = LocalEngine()
    X_L, X_D = engine.initialize(M_c, M_r, T, seed, n_chains=n_chains)

    return T, M_r, M_c, X_L, X_D, engine

def test_simple_predictive_probability_observed():
    # TODO
    pass

def test_simple_predictive_probability_unobserved(seed=0):
    T, M_r, M_c, X_L, X_D, engine = quick_le(seed)

    # Must query unobserved rows. CrossCat can only assess same row. We will
    # query one numerical column 0 and categorical column 2.
    Q = [[(N_ROWS, 0, 0.5)], [(N_ROWS, 2, 1)]]


    # Specify complex pattern of reasonable constraints. These tests demonstrate
    # desired behavior (no crashes), and were addressed in the branch
    # fsaad-conditional-pdf
    Y = [(0, 0, 1), (N_ROWS//2, 4, 5), (N_ROWS, 1, 0.5), (N_ROWS+1, 0, 1.2)]
    # - Numerical column Q[0].
    vals = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[0])
    # - Categorical column Q[1].
    vals = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[1])


    # XXX TODO: Fix more issues discovered
    # The next queries are probabilistically sound, but the code demonstrates
    # bizarre behavior. Determing what to do in all of these special cases is
    # a design problem.


    # Query the cell P(N_ROWS,0)=0.5 GIVEN (N_ROWS,0)==0.5
    # A reasonable human would expect this logp = log(1) = 0... guess not...

    # - Numerical cell Q[0].
    Y = [(N_ROWS, 0, 0.5)]
    val = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[0])
    with pytest.raises(AssertionError):
        assert val[0] == 0

    # - Categorical cell Q[1].
    Y = [(N_ROWS, 2, 1)]
    val = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[1])
    with pytest.raises(AssertionError):
        assert val[0] == 0


    # Query a hypothetical cell, constraining on another hypothetical row with
    # values in a DIFFERENT column as the column of the query cell.
    # This test does not cause a crash.

    # - Numerical cell Q[0]
    Y = [(N_ROWS, 0, 1.5), (N_ROWS+1, 1, 1.5)]
    vals = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[0])

    # - Categorical cell Q[1]
    Y = [(N_ROWS, 2, 4), (N_ROWS+1, 3, 5)]
    vals = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[1])


    # Query a hypothetical cell, constraining on another hypothetical row with
    # values in a SAME column as the column of the query cell.
    # This causes an IndexError ...

    # - Numerical cell Q[0]
    Y = [(N_ROWS, 0, 1.5), (N_ROWS+1, 0, 2)]
    with pytest.raises(IndexError):
        vals = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[0])

    # - Numerical cell Q[1]
    Y = [(N_ROWS, 2, 4), (N_ROWS+1, 2, 5)]
    with pytest.raises(IndexError):
        vals = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q[1])

def test_predictive_probability_observed(seed=0):
    # TODO
    pass

def test_predictive_probability_unobserved(seed=0):
    # This function tests the predictive probability for the joint distirbution.
    # Throughout, we will check that the result is the same for the joint and
    # simple calls.
    T, M_r, M_c, X_L, X_D, engine = quick_le(seed)

    # Hypothetical column number should throw an error.
    Q = [(N_ROWS, 1, 1.5), (N_ROWS, 10, 2)]
    Y = []
    with pytest.raises(ValueError):
        vals = engine.predictive_probability(M_c, X_L, X_D, Y, Q)

    # Inconsistent row numbers should throw an error.
    Q = [(N_ROWS, 1, 1.5), (N_ROWS-1, 10, 2)]
    Y = []
    with pytest.raises(ValueError):
        vals = engine.predictive_probability(M_c, X_L, X_D, Y, Q)

    # Duplicate column numbers should throw an error,
    Q = [(N_ROWS, 1, 1.5), (N_ROWS, 1, 2)]
    Y = []
    with pytest.raises(ValueError):
        val = engine.predictive_probability(M_c, X_L, X_D, Y, Q)

    # Different row numbers should throw an error.
    Q = [(N_ROWS, 0, 1.5), (N_ROWS+1, 1, 2)]
    Y = [(N_ROWS, 1, 1.5), (N_ROWS, 2, 3)]
    with pytest.raises(Exception):
        val = engine.predictive_probability(M_c, X_L, X_D, Y, Q[0])

    # Inconsistent with constraints should be negative infinity.
    Q = [(N_ROWS, 1, 1.5), (N_ROWS, 0, 1.3)]
    Y = [(N_ROWS, 1, 1.6)]
    val = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    assert val == -float('inf')
    assert isinstance(val, float)

    # Consistent with constraints should be log(1) == 0.
    Q = [(N_ROWS, 0, 1.3)]
    Y = [(N_ROWS, 0, 1.3)]
    val = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    assert val == 0

    # Consistent with constraints should not impact other queries.
    Q = [(N_ROWS, 1, 1.5), (N_ROWS, 0, 1.3)]
    Y = [(N_ROWS, 1, 1.5), (N_ROWS, 2, 3)]
    val_0 = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    val_1 = engine.predictive_probability(M_c, X_L, X_D, Y, Q[1:])
    assert val_0 == val_1

    # Predictive and simple should be the same in univariate case (cont).
    Q = [(N_ROWS, 0, 0.5)]
    Y = [(0, 0, 1), (N_ROWS//2, 4, 5), (N_ROWS, 1, 0.5), (N_ROWS+1, 0, 1.2)]
    val_0 = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    val_1 = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q)
    assert val_0 == val_1

    # Predictive and simple should be the same in univariate case (disc).
    Q = [(N_ROWS, 2, 1)]
    Y = [(0, 0, 1), (N_ROWS//2, 4, 5), (N_ROWS, 1, 0.5), (N_ROWS+1, 0, 1.2)]
    val_0 = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    val_1 = engine.simple_predictive_probability(M_c, X_L, X_D, Y, Q)
    assert val_0 == val_1

    # Do some full joint queries, all on the same row.
    Q = [(N_ROWS, 3, 4), (N_ROWS, 4, 1.3)]
    Y = [(N_ROWS, 0, 1), (N_ROWS, 1, -0.7), (N_ROWS, 2, 3)]
    val = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    assert isinstance(val, float)

    Q = [(N_ROWS, 0, 1), (N_ROWS, 1, -0.7), (N_ROWS, 2, 3)]
    Y = [(N_ROWS, 3, 4), (N_ROWS, 4, 1.3)]
    val = engine.predictive_probability(M_c, X_L, X_D, Y, Q)
    assert isinstance(val, float)
