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
from six.moves import range
import sys
import csv
import copy
import random
#
import numpy


def get_generative_clustering(seed, M_c, M_r, T,
                              data_inverse_permutation_indices,
                              num_clusters, num_views):
    from crosscat.LocalEngine import LocalEngine
    import crosscat.cython_code.State as State
    rng = random.Random(seed)
    get_next_seed = lambda: rng.randint(1, 2**31 - 1)
    # NOTE: this function only works because State.p_State doesn't use
    #       column_component_suffstats
    num_rows = len(T)
    num_cols = len(T[0])
    X_D_helper = numpy.repeat(range(num_clusters), (num_rows / num_clusters))
    gen_X_D = [
        X_D_helper[numpy.argsort(data_inverse_permutation_index)]
        for data_inverse_permutation_index in data_inverse_permutation_indices
        ]
    gen_X_L_assignments = numpy.repeat(range(num_views), (num_cols / num_views))
    # initialize to generate an X_L to manipulate
    local_engine = LocalEngine()
    bad_X_L, bad_X_D = local_engine.initialize(M_c, M_r, T, get_next_seed(),
                                                         initialization='apart')
    bad_X_L['column_partition']['assignments'] = gen_X_L_assignments
    # manually constrcut state in in generative configuration
    state = State.p_State(M_c, T, bad_X_L, gen_X_D)
    gen_X_L = state.get_X_L()
    gen_X_D = state.get_X_D()
    # run inference on hyperparameters to leave them in a reasonable state
    kernel_list = (
        'row_partition_hyperparameters',
        'column_hyperparameters',
        'column_partition_hyperparameter',
        )
    gen_X_L, gen_X_D = local_engine.analyze(M_c, T, gen_X_L, gen_X_D,
            get_next_seed(), n_steps=1, kernel_list=kernel_list)
    #
    return gen_X_L, gen_X_D

def generate_clean_state(gen_seed, num_clusters,
                         num_cols, num_rows, num_splits,
                         max_mean=10, max_std=1,
                         plot=False):
    rng = random.Random(gen_seed)
    get_next_seed = lambda: rng.randint(1, 2**31 - 1)
    # generate the data
    T, M_r, M_c, data_inverse_permutation_indices = \
        gen_factorial_data_objects(get_next_seed(), num_clusters,
                                      num_cols, num_rows, num_splits,
                                      max_mean=10, max_std=1,
                                      send_data_inverse_permutation_indices=True)
    # recover generative clustering
    X_L, X_D = get_generative_clustering(get_next_seed(), M_c, M_r, T,
                                         data_inverse_permutation_indices,
                                         num_clusters, num_splits)
    return T, M_c, M_r, X_L, X_D

def get_ith_ordering(in_list, i):
    temp_list = [in_list[j::(i+1)][:] for j in range(i+1)]
    return [el for sub_list in temp_list for el in sub_list]

def gen_data(gen_seed, num_clusters,
             num_cols, num_rows, max_mean_per_category=10, max_std=1,
             max_mean=None):
    if max_mean is None:
       max_mean = max_mean_per_category * num_clusters
    n_grid = 11
    mu_grid = numpy.linspace(-max_mean, max_mean, n_grid)
    sigma_grid = 10 ** numpy.linspace(-1, numpy.log10(max_std), n_grid)
    num_rows_per_cluster = num_rows / num_clusters
    zs = numpy.repeat(range(num_clusters), num_rows_per_cluster)
    #
    random_state = numpy.random.RandomState(gen_seed)
    #
    data_size = (num_clusters,num_cols)
    which_mus = random_state.randint(len(mu_grid), size=data_size)
    which_sigmas = random_state.randint(len(sigma_grid), size=data_size)
    mus = mu_grid[which_mus]
    sigmas = sigma_grid[which_sigmas]
    clusters = []
    for row_mus, row_sigmas in zip(mus, sigmas):
        cluster_columns = []
        for mu, sigma in zip(row_mus, row_sigmas):
            cluster_column = random_state.normal(mu, sigma,
                                                 num_rows_per_cluster)
            cluster_columns.append(cluster_column)
        cluster = numpy.vstack(cluster_columns).T
        clusters.append(cluster)
    xs = numpy.vstack(clusters)
    return xs, zs

def gen_factorial_data(gen_seed, num_clusters,
        num_cols, num_rows, num_splits,
		max_mean_per_category=10, max_std=1,
        max_mean=None
        ):
    random_state = numpy.random.RandomState(gen_seed)
    data_list = []
    inverse_permutation_indices_list = []
    for data_idx in range(num_splits):
        data_i, zs_i = gen_data(
            gen_seed=gen_seed,
            num_clusters=num_clusters,
            num_cols=num_cols/num_splits,
            num_rows=num_rows,
            max_mean_per_category=max_mean_per_category,
            max_std=max_std,
            max_mean=max_mean
            )
        permutation_indices = random_state.permutation(range(num_rows))
        # permutation_indices = get_ith_ordering(range(num_rows), data_idx)
        inverse_permutation_indices = numpy.argsort(permutation_indices)
        inverse_permutation_indices_list.append(inverse_permutation_indices)
        data_list.append(numpy.array(data_i)[permutation_indices])
    data = numpy.hstack(data_list)
    return data, inverse_permutation_indices_list

def gen_M_r_from_T(T):
    num_rows = len(T)
    num_cols = len(T[0])
    #
    name_to_idx = dict(zip(map(str, range(num_rows)), range(num_rows)))
    idx_to_name = dict(zip(map(str, range(num_rows)), range(num_rows)))
    M_r = dict(name_to_idx=name_to_idx, idx_to_name=idx_to_name)
    return M_r

def gen_continuous_metadata(column_data):
    return dict(
        modeltype="normal_inverse_gamma",
        value_to_code=dict(),
        code_to_value=dict(),
        )

def gen_cyclic_metadata(column_data):
    return dict(
        modeltype="vonmises",
        value_to_code=dict(),
        code_to_value=dict(),
        )

def gen_multinomial_metadata(column_data):
    def get_is_not_nan(el):
        if isinstance(el, str):
            return el.upper() != 'NAN'
        else:
            return True
    # get_is_not_nan = lambda el: el.upper() != 'NAN'
    #
    unique_codes = [el for el in set(column_data) if get_is_not_nan(el)]
    #
    values = list(range(len(unique_codes)))
    value_to_code = dict(zip(values, unique_codes))
    code_to_value = dict(zip(unique_codes, values))
    return dict(
        modeltype="symmetric_dirichlet_discrete",
        value_to_code=value_to_code,
        code_to_value=code_to_value,
        )

metadata_generator_lookup = dict(
    continuous=gen_continuous_metadata,
    multinomial=gen_multinomial_metadata,
    cyclic=gen_cyclic_metadata,
)

def gen_M_c_from_T(T, cctypes=None, colnames=None):
    num_rows = len(T)
    num_cols = len(T[0])
    if cctypes is None:
        cctypes = ['continuous'] * num_cols
    if colnames is None:
        colnames = list(range(num_cols))
    #
    T_array_transpose = numpy.array(T).T
    column_metadata = []
    for cctype, column_data in zip(cctypes, T_array_transpose):
        metadata_generator = metadata_generator_lookup[cctype]
        metadata = metadata_generator(column_data)
        column_metadata.append(metadata)
    name_to_idx = dict(zip(colnames, range(num_cols)))
    idx_to_name = dict(zip(map(str, range(num_cols)), colnames))
    M_c = dict(
        name_to_idx=name_to_idx,
        idx_to_name=idx_to_name,
        column_metadata=column_metadata,
        )
    return M_c

def gen_M_c_from_T_with_colnames(T, colnames):
    num_rows = len(T)
    num_cols = len(T[0])
    #
    gen_continuous_metadata = lambda: dict(modeltype="normal_inverse_gamma",
                                           value_to_code=dict(),
                                           code_to_value=dict())
    column_metadata = [
        gen_continuous_metadata()
        for col_idx in range(num_cols)
        ]
    name_to_idx = dict(zip(colnames, range(num_cols)))
    idx_to_name = dict(zip(map(str, range(num_cols)),colnames))
    M_c = dict(
        name_to_idx=name_to_idx,
        idx_to_name=idx_to_name,
        column_metadata=column_metadata,
        )
    return M_c

def gen_factorial_data_objects(gen_seed, num_clusters,
                               num_cols, num_rows, num_splits,
                               max_mean=10, max_std=1,
                               send_data_inverse_permutation_indices=False):
    T, data_inverse_permutation_indices = gen_factorial_data(
        gen_seed, num_clusters,
        num_cols, num_rows, num_splits, max_mean, max_std)
    T  = T.tolist()
    M_r = gen_M_r_from_T(T)
    M_c = gen_M_c_from_T(T)
    if not send_data_inverse_permutation_indices:
        return T, M_r, M_c
    else:
        return T, M_r, M_c, data_inverse_permutation_indices

def discretize_data(T, discretize_indices):
    T_array = numpy.array(T)
    discretize_indices = numpy.array(discretize_indices)
    T_array[:, discretize_indices] = \
        numpy.array(T_array[:, discretize_indices], dtype=int)
    return T_array.tolist()

def convert_columns_to_multinomial(T, M_c, multinomial_indices):
    multinomial_indices = numpy.array(multinomial_indices)
    modeltype = 'symmetric_dirichlet_discrete'
    T_array = numpy.array(T)
    for multinomial_idx in multinomial_indices:
        multinomial_column = T_array[:, multinomial_idx]
        multinomial_column = multinomial_column[~numpy.isnan(multinomial_column)]
        multinomial_values = list(set(multinomial_column))
        K = len(multinomial_values)
        code_to_value = dict(zip(range(K), multinomial_values))
        value_to_code = dict(zip(multinomial_values, range(K)))
        multinomial_column_metadata = M_c['column_metadata'][multinomial_idx]
        multinomial_column_metadata['modeltype'] = modeltype
        multinomial_column_metadata['code_to_value'] = code_to_value
        multinomial_column_metadata['value_to_code'] = value_to_code
    return T, M_c

# UNTESTED
def convert_columns_to_continuous(T, M_c, continuous_indices):
    continuous_indices = numpy.array(continuous_indices)
    modeltype = 'normal_inverse_gamma'
    T_array = numpy.array(T)
    for continuous_idx in continuous_indices:
        code_to_value = dict()
        value_to_code = dict()
        continuous_column_metadata = M_c['column_metadata'][continuous_idx]
        continuous_column_metadata['modeltype'] = modeltype
        continuous_column_metadata['code_to_value'] = code_to_value
        continuous_column_metadata['value_to_code'] = value_to_code
    return T, M_c

def at_most_N_rows(T, N, gen_seed=0):
    num_rows = len(T)
    if (N is not None) and (num_rows > N):
        random_state = numpy.random.RandomState(gen_seed)
        which_rows = random_state.permutation(range(num_rows))
        which_rows = which_rows[:N]
        T = [T[which_row] for which_row in which_rows]
    return T

def read_csv(filename, has_header=True):
    with open(filename) as fh:
        csv_reader = csv.reader(fh)
        header = None
        if has_header:
            header = csv_reader.next()
        rows = [row for row in csv_reader]
    return header, rows

def write_csv(filename, T, header = None):
    with open(filename,'w') as fh:
        csv_writer = csv.writer(fh, delimiter=',')
        if header != None:
            csv_writer.writerow(header)
        [csv_writer.writerow(T[i]) for i in range(len(T))]

def all_continuous_from_file(filename, max_rows=None, gen_seed=0, has_header=True):
    header, T = read_csv(filename, has_header=has_header)
    T = numpy.array(T, dtype=float).tolist()
    T = at_most_N_rows(T, N=max_rows, gen_seed=gen_seed)
    M_r = gen_M_r_from_T(T)
    M_c = gen_M_c_from_T(T)
    return T, M_r, M_c, header

def continuous_or_ignore_from_file_with_colnames(filename, cctypes, max_rows=None, gen_seed=0):
    header = None
    T, M_r, M_c = None, None, None
    colmask = map(lambda x: 1 if x != 'ignore' else 0, cctypes)
    with open(filename) as fh:
        csv_reader = csv.reader(fh)
        header = csv_reader.next()
        T = numpy.array([
                [col for col, flag in zip(row, colmask) if flag] for row in csv_reader
                ], dtype=float).tolist()
        num_rows = len(T)
        if (max_rows is not None) and (num_rows > max_rows):
            random_state = numpy.random.RandomState(gen_seed)
            which_rows = random_state.permutation(range(num_rows))
            which_rows = which_rows[:max_rows]
            T = [T[which_row] for which_row in which_rows]
        M_r = gen_M_r_from_T(T)
        M_c = gen_M_c_from_T_with_colnames(T, [col for col, flag in zip(header, colmask) if flag])
    return T, M_r, M_c, header

def convert_code_to_value(M_c, cidx, code):
    """
    For a column with categorical data, this function takes the 'code':
    the integer used to represent a specific value, and returns the corresponding
    raw value (e.g. 'Joe' or 234.23409), which is always encoded as a string.

    Note that the underlying store 'value_to_code' is unfortunately named backwards.
    TODO: fix the backwards naming.
    """
    if M_c['column_metadata'][cidx]['modeltype'] == 'normal_inverse_gamma':
        return float(code)
    else:
        try:
            return M_c['column_metadata'][cidx]['value_to_code'][int(code)]
        except KeyError:
            return M_c['column_metadata'][cidx]['value_to_code'][str(int(code))]

def convert_value_to_code(M_c, cidx, value):
    """
    For a column with categorical data, this function takes the raw value
    (e.g. 'Joe' or 234.23409), which is always encoded as a string, and returns the
    'code': the integer used to represent that value in the underlying representation.

    Note that the underlying store 'code_to_value' is unfortunately named backwards.
    TODO: fix the backwards naming.
    """
    if M_c['column_metadata'][cidx]['modeltype'] == 'normal_inverse_gamma':
        return float(value)
    else:
        return M_c['column_metadata'][cidx]['code_to_value'][str(value)] 

def map_from_T_with_M_c(coordinate_value_tuples, M_c):
    coordinate_code_tuples = []
    column_metadata = M_c['column_metadata']
    for row_idx, col_idx, value in coordinate_value_tuples:
        datatype = column_metadata[col_idx]['modeltype']
        # FIXME: make this robust to different datatypes
        if datatype == 'symmetric_dirichlet_discrete':
            # FIXME: casting key to str is a hack
            value = column_metadata[col_idx]['value_to_code'][str(int(value))]
        coordinate_code_tuples.append((row_idx, col_idx, value))
    return coordinate_code_tuples

def map_to_T_with_M_c(T_uncast_array, M_c):
    T_uncast_array = numpy.array(T_uncast_array)
    # WARNING: array argument is mutated
    for col_idx in range(T_uncast_array.shape[1]):
        modeltype = M_c['column_metadata'][col_idx]['modeltype']
        if modeltype != 'symmetric_dirichlet_discrete': continue
        # copy.copy else you mutate M_c
        mapping = copy.copy(M_c['column_metadata'][col_idx]['code_to_value'])
        mapping['NAN'] = numpy.nan
        col_data = T_uncast_array[:, col_idx]
        to_upper = lambda el: el.upper()
        is_nan_str = numpy.array(map(to_upper, col_data))=='NAN'
        col_data[is_nan_str] = 'NAN'
        # FIXME: THIS IS WHERE TO PUT NAN HANDLING
        mapped_values = [mapping[el] for el in col_data]
        T_uncast_array[:, col_idx] = mapped_values
    T = numpy.array(T_uncast_array, dtype=float).tolist()
    return T

def do_pop_list_indices(in_list, pop_indices):
    pop_indices = sorted(pop_indices, reverse=True)
    _do_pop = lambda x: in_list.pop(x)
    map(_do_pop, pop_indices)
    return in_list

def get_list_indices(in_list, get_indices_of):
    lookup = dict(zip(in_list, range(len(in_list))))
    indices = map(lookup.get, get_indices_of)
    indices = filter(None, indices)
    return indices

def transpose_list(in_list):
    return zip(*in_list)

def get_pop_indices(cctypes, colnames):
    assert len(colnames) == len(cctypes)
    pop_columns = [
            colname
            for (cctype, colname) in zip(cctypes, colnames)
            if cctype == 'ignore'
            ]
    pop_indices = get_list_indices(colnames, pop_columns)
    return pop_indices

def do_pop_columns(T, pop_indices):
    T_by_columns = transpose_list(T)
    T_by_columns = do_pop_list_indices(T_by_columns, pop_indices)
    T = transpose_list(T_by_columns)
    return T

def remove_ignore_cols(T, cctypes, colnames):
    pop_indices = get_pop_indices(cctypes, colnames)
    T = do_pop_columns(T, pop_indices)
    colnames = do_pop_list_indices(colnames[:], pop_indices)
    cctypes = do_pop_list_indices(cctypes[:], pop_indices)
    return T, cctypes, colnames

nan_set = set(['', 'null', 'n/a'])
_convert_nan = lambda el: el if el.strip().lower() not in nan_set else 'NAN'
_convert_nans = lambda in_list: map(_convert_nan, in_list)
convert_nans = lambda in_T: map(_convert_nans, in_T)

def read_data_objects(filename, max_rows=None, gen_seed=0,
                      cctypes=None, colnames=None):
    header, raw_T = read_csv(filename, has_header=True)
    header = [h.lower().strip() for h in header]
    # FIXME: why both accept colnames argument and read header?
    if colnames is None:
        colnames = header
        pass
    # remove excess rows
    raw_T = at_most_N_rows(raw_T, N=max_rows, gen_seed=gen_seed)
    raw_T = convert_nans(raw_T)
    # remove ignore columns
    if cctypes is None:
        cctypes = ['continuous'] * len(header)
        pass
    T_uncast_arr, cctypes, header = remove_ignore_cols(raw_T, cctypes, header)
    # determine value mappings and map T to continuous castable values
    M_r = gen_M_r_from_T(T_uncast_arr)
    M_c = gen_M_c_from_T(T_uncast_arr, cctypes, colnames)
    T = map_to_T_with_M_c(T_uncast_arr, M_c)
    #
    return T, M_r, M_c, header

def get_can_cast_to_float(column_data):
    can_cast = True
    try:
        [float(datum) for datum in column_data]
    except ValueError as e:
        can_cast = False
    return can_cast
    
def guess_column_type(column_data, count_cutoff=20, ratio_cutoff=0.02):
    num_distinct = len(set(column_data))
    num_data = len(column_data)
    distinct_ratio = float(num_distinct) / num_data
    above_count_cutoff = num_distinct > count_cutoff
    above_ratio_cutoff = distinct_ratio > ratio_cutoff
    can_cast = get_can_cast_to_float(column_data)
    if above_count_cutoff and above_ratio_cutoff and can_cast:
        column_type = 'continuous'
    else:
        column_type = 'multinomial'
    return column_type

def guess_column_types(T, count_cutoff=20, ratio_cutoff=0.02):
    T_transposed = transpose_list(T)
    column_types = []
    for column_data in T_transposed:
        column_type = guess_column_type(column_data, count_cutoff, ratio_cutoff)
        column_types.append(column_type)
    return column_types
        
def read_model_data_from_csv(filename, max_rows=None, gen_seed=0,
                             cctypes=None):
    colnames, T = read_csv(filename)
    T = at_most_N_rows(T, max_rows, gen_seed)
    T = convert_nans(T)
    if cctypes is None:
        cctypes = guess_column_types(T)
    M_c = gen_M_c_from_T(T, cctypes, colnames)
    T = map_to_T_with_M_c(numpy.array(T), M_c)
    M_r = gen_M_r_from_T(T)
    return T, M_r, M_c

extract_view_count = lambda X_L: len(X_L['view_state'])
extract_cluster_count = lambda view_state_i: view_state_i['row_partition_model']['counts']
extract_cluster_counts = lambda X_L: map(extract_cluster_count, X_L['view_state'])
get_state_shape = lambda X_L: (extract_view_count(X_L), extract_cluster_counts(X_L))
