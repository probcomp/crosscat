from crosscat import LocalEngine as LE
from crosscat.utils import data_utils as du
import random
import time
import pytest
import itertools as it

N_COLS = 10
N_ROWS = 10


def first_duplicate(a):
    """ Returns the indices of the first duplicated item in the list, a. """
    seen = set([a[0]])
    for item in a[1:]:
        if item not in seen:
            seen.add(item)
        else:
            return [i for i in range(len(a)) if a[i] == item]
    return None


@pytest.fixture
def quick_le():
    n_chains = 1
    seed = int(time.time())
    T, M_r, M_c = du.gen_factorial_data_objects(seed, 2, N_COLS, N_ROWS, 2)
    engine = LE.LocalEngine(seed=seed)
    X_L, X_D = engine.initialize(M_c, M_r, T, n_chains=n_chains)
    return T, M_r, M_c, X_L, X_D, engine


def test_one_col_dependency_init(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    while len(set(X_L['column_partition']['assignments'])) == 1:
        X_L, X_D = engine.initialize(M_c, M_r, T)

    col1 = X_L['column_partition']['assignments'].index(0)
    col2 = X_L['column_partition']['assignments'].index(1)

    assert X_L['column_partition']['assignments'][col1] != \
        X_L['column_partition']['assignments'][col2]

    relinfo = [(col1, col2, True)]
    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    assert engine.assert_col(X_L, X_D, col1, col2, True, True)


def test_one_col_independency_init(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    while len(set(X_L['column_partition']['assignments'])) == N_COLS:
        X_L, X_D = engine.initialize(M_c, M_r, T)

    dup_idx = first_duplicate(X_L['column_partition']['assignments'])
    col1 = dup_idx[0]
    col2 = dup_idx[1]

    assert X_L['column_partition']['assignments'][col1] == \
        X_L['column_partition']['assignments'][col2]

    relinfo = [(col1, col2, False)]
    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    assert engine.assert_col(X_L, X_D, col1, col2, False, True)


def test_one_of_each_init(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    colrange = [i for i in range(N_COLS)]

    dep_col1 = random.choice(colrange)
    del colrange[colrange.index(dep_col1)]
    dep_col2 = random.choice(colrange)

    ind_col1 = random.choice(colrange)
    del colrange[colrange.index(ind_col1)]
    ind_col2 = random.choice(colrange)

    relinfo = [(dep_col1, dep_col2, True), (ind_col1, ind_col2, False)]
    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    for col1, col2, dependent in relinfo:
        assert engine.assert_col(X_L, X_D, col1, col2, dependent, True)


def test_one_col_dependency_analyze(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    while len(set(X_L['column_partition']['assignments'])) == 1:
        X_L, X_D = engine.initialize(M_c, M_r, T)

    col1 = X_L['column_partition']['assignments'].index(0)
    col2 = X_L['column_partition']['assignments'].index(1)

    assert X_L['column_partition']['assignments'][col1] != \
        X_L['column_partition']['assignments'][col2]

    relinfo = [(col1, col2, True)]
    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    assert engine.assert_col(X_L, X_D, col1, col2, True, True)

    X_L, X_D = engine.analyze(M_c, T, X_L, X_D, n_steps=1)

    assert engine.assert_col(X_L, X_D, col1, col2, True, True)


def test_one_col_independency_analyze(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    while len(set(X_L['column_partition']['assignments'])) == N_COLS:
        X_L, X_D = engine.initialize(M_c, M_r, T)

    dup_idx = first_duplicate(X_L['column_partition']['assignments'])
    col1 = dup_idx[0]
    col2 = dup_idx[1]

    assert X_L['column_partition']['assignments'][col1] == \
        X_L['column_partition']['assignments'][col2]

    relinfo = [(col1, col2, False)]
    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    assert engine.assert_col(X_L, X_D, col1, col2, False, True)

    X_L, X_D = engine.analyze(M_c, T, X_L, X_D, n_steps=1)

    assert engine.assert_col(X_L, X_D, col1, col2, False, True)


def test_multiple_col_dependency_analyze(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    while len(set(X_L['column_partition']['assignments'])) == N_COLS:
        X_L, X_D = engine.initialize(M_c, M_r, T)
    col_pairs = [(c1, c2,) for c1, c2 in it.combinations(range(N_COLS), 2)]
    independent_pairs = random.sample(col_pairs, 4)
    relinfo = [(c1, c2, True) for c1, c2 in independent_pairs]

    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    for col1, col2, dep in relinfo:
        assert engine.assert_col(X_L, X_D, col1, col2, dep, True)

    X_L, X_D = engine.analyze(M_c, T, X_L, X_D, n_steps=1)

    for col1, col2, dep in relinfo:
        assert engine.assert_col(X_L, X_D, col1, col2, dep, True)


def test_multiple_col_independency_analyze(quick_le):
    T, M_r, M_c, X_L, X_D, engine = quick_le
    while len(set(X_L['column_partition']['assignments'])) == N_COLS:
        X_L, X_D = engine.initialize(M_c, M_r, T)
    col_pairs = [(c1, c2,) for c1, c2 in it.combinations(range(N_COLS), 2)]
    independent_pairs = random.sample(col_pairs, 4)
    relinfo = [(c1, c2, False) for c1, c2 in independent_pairs]

    X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)

    for col1, col2, dep in relinfo:
        assert engine.assert_col(X_L, X_D, col1, col2, dep, True)

    X_L, X_D = engine.analyze(M_c, T, X_L, X_D, n_steps=1)

    for col1, col2, dep in relinfo:
        assert engine.assert_col(X_L, X_D, col1, col2, dep, True)


# # FIXME: causes segfault
# def test_one_of_each_analyze(quick_le):
#     T, M_r, M_c, X_L, X_D, engine = quick_le
#     colrange = [i for i in range(N_COLS)]
# 
#     dep_col1 = random.choice(colrange)
#     del colrange[colrange.index(dep_col1)]
#     dep_col2 = random.choice(colrange)
# 
#     ind_col1 = random.choice(colrange)
#     del colrange[colrange.index(ind_col1)]
#     ind_col2 = random.choice(colrange)
# 
#     relinfo = [(dep_col1, dep_col2, True), (ind_col1, ind_col2, False)]
#     X_L, X_D = engine.ensure_col(M_c, M_r, T, X_L, X_D, relinfo)
# 
#     for col1, col2, dependent in relinfo:
#         assert engine.assert_col(X_L, X_D, col1, col2, dependent, True)
# 
#     X_L, X_D = engine.analyze(M_c, T, X_L, X_D, n_steps=1)
# 
#     for col1, col2, dependent in relinfo:
#         assert engine.assert_col(X_L, X_D, col1, col2, dependent, True)
