from crosscat import LocalEngine as LE
from crosscat.utils import data_utils as du
import random
import numpy
import pytest
import itertools as it

N_COLS = 10
N_ROWS = 10

FAILURE_SEED = 130223
PASS_SEED = 67944

get_next_seed = lambda rng: rng.randint(1, 2**31 - 1)

single_args = [item for item in it.product(
    [FAILURE_SEED, PASS_SEED], [True, False], [True, False])]
one_each_args = [item for item in it.product(
    [FAILURE_SEED, PASS_SEED], [True, False])]


def first_duplicate(a):
    """ Returns the indices of the first duplicated item in the list, a. """
    seen = set([a[0]])
    for item in a[1:]:
        if item not in seen:
            seen.add(item)
        else:
            return [i for i in range(len(a)) if a[i] == item]
    return None


def quick_le(seed, n_chains=1):
    rng = random.Random(seed)
    T, M_r, M_c = du.gen_factorial_data_objects(get_next_seed(rng), 2,
        N_COLS, N_ROWS, 2)
    engine = LE.LocalEngine(seed=get_next_seed(rng))
    X_L, X_D = engine.initialize(M_c, M_r, T, seed=get_next_seed(rng),
        n_chains=n_chains)
    return T, M_r, M_c, X_L, X_D, engine


@pytest.mark.parametrize("seed, dependent, analyze", single_args)
def test_col_ensure_with_single_constraint(seed, dependent, analyze):
    rng = random.Random(seed)
    T, M_r, M_c, X_L, X_D, engine = quick_le(get_next_seed(rng))
    if dependent:
        while len(set(X_L['column_partition']['assignments'])) == 1:
            X_L, X_D = engine.initialize(M_c, M_r, T, get_next_seed(rng))
        col1 = X_L['column_partition']['assignments'].index(0)
        col2 = X_L['column_partition']['assignments'].index(1)
        assert X_L['column_partition']['assignments'][col1] != \
            X_L['column_partition']['assignments'][col2]
    else:
        while len(set(X_L['column_partition']['assignments'])) == N_COLS:
            X_L, X_D = engine.initialize(M_c, M_r, T, get_next_seed(rng))
        dup_idx = first_duplicate(X_L['column_partition']['assignments'])
        col1 = dup_idx[0]
        col2 = dup_idx[1]
        assert X_L['column_partition']['assignments'][col1] == \
            X_L['column_partition']['assignments'][col2]

    dep_constraints = [(col1, col2, dependent)]
    X_L, X_D = engine.ensure_col_dep_constraints(M_c, M_r, T, X_L, X_D,
        dep_constraints, get_next_seed(rng))

    assert engine.assert_col_dep_constraints(X_L, X_D, col1, col2, dependent, True)

    if analyze:
        X_L, X_D = engine.analyze(M_c, T, X_L, X_D, get_next_seed(rng),
            n_steps=1)
        assert engine.assert_col_dep_constraints(X_L, X_D, col1, col2, dependent, True)


@pytest.mark.parametrize("seed, analyze", one_each_args)
def test_one_of_each_init(seed, analyze):
    rng = random.Random(seed)
    T, M_r, M_c, X_L, X_D, engine = quick_le(get_next_seed(rng))
    colrange = [i for i in range(N_COLS)]

    dep_col1 = random.choice(colrange)
    del colrange[colrange.index(dep_col1)]
    dep_col2 = random.choice(colrange)

    ind_col1 = random.choice(colrange)
    del colrange[colrange.index(ind_col1)]
    ind_col2 = random.choice(colrange)

    dep_constraints = [(dep_col1, dep_col2, True), (ind_col1, ind_col2, False)]
    X_L, X_D = engine.ensure_col_dep_constraints(M_c, M_r, T, X_L, X_D,
        dep_constraints, get_next_seed(rng))

    for col1, col2, dependent in dep_constraints:
        assert engine.assert_col_dep_constraints(X_L, X_D, col1, col2, dependent, True)

    if analyze:
        X_L, X_D = engine.analyze(M_c, T, X_L, X_D, get_next_seed(rng),
            n_steps=1)

        for col1, col2, dependent in dep_constraints:
            assert engine.assert_col_dep_constraints(X_L, X_D, col1, col2, dependent, True)


@pytest.mark.parametrize("seed, dependent, analyze", single_args)
def test_multiple_col_ensure(seed, dependent, analyze):
    rng = random.Random(seed)
    T, M_r, M_c, X_L, X_D, engine = quick_le(get_next_seed(rng))
    col_pairs = [(c1, c2,) for c1, c2 in it.combinations(range(N_COLS), 2)]
    ensure_pairs = random.sample(col_pairs, 4)
    dep_constraints = [(c1, c2, dependent) for c1, c2 in ensure_pairs]

    X_L, X_D = engine.ensure_col_dep_constraints(M_c, M_r, T, X_L, X_D,
        dep_constraints, get_next_seed(rng))

    for col1, col2, dep in dep_constraints:
        assert engine.assert_col_dep_constraints(X_L, X_D, col1, col2, dep, True)

    if analyze:
        X_L, X_D = engine.analyze(M_c, T, X_L, X_D, get_next_seed(rng),
            n_steps=1000)

        for col1, col2, dep in dep_constraints:
            assert engine.assert_col_dep_constraints(X_L, X_D, col1, col2, dep, True)

def test_two_dependent_one_dependent():
    rng = random.Random(PASS_SEED)
    T, M_r, M_c = du.gen_factorial_data_objects(get_next_seed(rng), 2, 3, 10, 1)
    engine = LE.LocalEngine(seed=get_next_seed(rng))

    # These dependency constraints target the computation of unorm_crp_logps_avg
    # in the case that one variable (1) is independent of another variable (2)
    # which should ensure the CRP probability of the view of (2) is -INFINITY
    # when doing a block proposal of (0,1) into the view of (2). Refer to the
    # comment about compute the CRP probabilities in
    # State::sample_insert_features.
    dep_constraints = [(0, 1, True), (1, 2, False)]

    X_L, X_D = engine.initialize(
        M_c, M_r, T, seed=get_next_seed(rng), n_chains=1)

    X_L, X_D = engine.ensure_col_dep_constraints(
        M_c, M_r, T, X_L, X_D, dep_constraints, get_next_seed(rng))

    for col1, col2, dep in dep_constraints:
        assert engine.assert_col_dep_constraints(
            X_L, X_D, col1, col2, dep, True)

    X_L, X_D = engine.analyze(
        M_c, T, X_L, X_D, get_next_seed(rng), n_steps=1000)

    for col1, col2, dep in dep_constraints:
        assert engine.assert_col_dep_constraints(
            X_L, X_D, col1, col2, dep, True)
