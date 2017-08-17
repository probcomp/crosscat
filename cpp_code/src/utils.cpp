/*
*   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
*
*   Lead Developers: Dan Lovell and Jay Baxter
*   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
*   Research Leads: Vikash Mansinghka, Patrick Shafto
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.
*/
#include "RandomNumberGenerator.h"
#include "utils.h"
//
#include <numeric>
#include <algorithm>

using namespace std;

bool is_almost(double val1, double val2, double precision)
{
    return abs(val1 - val2) < precision;
}

// linspace(a, b, n)
//
//      Return a vector of n equidistant points between a and b,
//      inclusive, for n >= 2.  That is,
//
//              [a, a + s, a + 2s, a + 3s, ..., a + (n - 2)s, b],
//
//      where s = (b - a)/(n - 1) is the distance between consecutive
//      points.  If v is the result, then v[0] = a and v[n - 1] = b.
vector<double> linspace(double a, double b, size_t n)
{
    assert(2 <= n);
    assert(a <= b);
    vector<double> v(n);
    const double s = (b - a) / (n - 1);
    v[0] = a;
    for (size_t i = 1; i < (n - 1); i++) {
        v[i] = a + i * s;
    }
    v[n - 1] = b;
    return v;
}

// log_linspace(a, b, n)
//
//      Return a vector of n `equiratioed' points between a and b,
//      inclusive, for n >= 2.  That is,
//
//              [a, a s, a s^2, a s^3, ..., a s^(n - 2), b],
//
//      where s = (b/a)^1/(n - 1) is the ratio of consecutive points.
//      If v is the result, then v[0] = a and v[n - 1] = b, except in
//      the case described below.
//
//      If a or b is too close to zero to be represented normally, it
//      is first rounded up to the smallest positive normal value.
//      Analytically, zero makes no sense for either a or b, but
//      values sufficiently close to zero will have been rounded to
//      zero beforehand.
//
//      We ignore subnormals because starting your computation with
//      them is unlikely to help you but quite likely to slow you
//      down.
vector<double> log_linspace(double a, double b, size_t n)
{
    assert(2 <= n);
    assert(a <= b);
    a = std::max(a, std::numeric_limits<double>::min());
    b = std::max(b, std::numeric_limits<double>::min());
    vector<double> v(n);
    const double log_a = log(a);
    const double log_r = (log(b) - log_a) / (n - 1);
    v[0] = a;
    for (size_t i = 1; i < (n - 1); i++) {
        v[i] = std::max(a, std::min(b, exp(log_a + i * log_r)));
    }
    v[n - 1] = b;
    return v;
}

vector<double> std_vector_divide_elemwise(
    const vector<double> &vec,
    const double &val)
{
    vector<double> result;
    for (size_t i = 0; i < vec.size(); i++) {
        result.push_back(vec[i]/val);
    }
    return result;
}

vector<double> std_vector_add(const vector<double> &vec1,
    const vector<double> &vec2)
{
    assert(vec1.size() == vec2.size());
    vector<double> sum_vec;
    for (unsigned int i = 0; i < vec1.size(); i++) {
        sum_vec.push_back(vec1[i] + vec2[i]);
    }
    return sum_vec;
}

vector<double> std_vector_add(const vector<vector<double> > &vec_vec)
{
    vector<double> sum_vec = vec_vec[0];
    vector<vector<double> >::const_iterator it = vec_vec.begin();
    it++;
    for (; it != vec_vec.end(); it++) {
        sum_vec = std_vector_add(sum_vec, *it);
    }
    return sum_vec;
}

static vector<double> filter_nans(const vector<double> &values)
{
    vector<double> non_nan_values;
    vector<double>::const_iterator it;
    for (it = values.begin(); it != values.end(); it++) {
        if (isnan(*it)) {
            continue;
        }
        non_nan_values.push_back(*it);
    }
    return non_nan_values;
}

double std_vector_sum(const vector<double> &values)
{
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum;
}

double std_vector_mean(const vector<double> &values)
{
    double sum = std_vector_sum(values);
    double mean = sum / values.size();
    return mean;
}

double calc_sum_sq_deviation(const vector<double> &values)
{
    double mean = std_vector_mean(values);
    double sum_sq_deviation = 0;
    vector<double>::const_iterator it;
    for (it = values.begin(); it != values.end(); it++) {
        sum_sq_deviation += pow((*it) - mean, 2) ;
    }
    return sum_sq_deviation;
}

vector<double> extract_row(const matrix<double> &data, int row_idx)
{
    vector<double> row;
    for (unsigned int j = 0; j < data.size2(); j++) {
        row.push_back(data(row_idx, j));
    }
    return row;
}

vector<double> extract_col(const matrix<double> &data, int col_idx)
{
    vector<double> col;
    for (unsigned int j = 0; j < data.size1(); j++) {
        col.push_back(data(j, col_idx));
    }
    return col;
}

vector<vector<double> > extract_cols(
    const matrix<double> &data, vector<int> &col_idxs)
{
    vector<vector<double> > cols;
    vector<int>::const_iterator it;
    for (it = col_idxs.begin(); it != col_idxs.end(); it++) {
        vector<double> col = extract_col(data, *it);
        cols.push_back(col);
    }
    return cols;
}

vector<int> extract_global_ordering(const map<int, int> &global_to_local)
{
    vector<int> global_indices(global_to_local.size(), -1);
    map<int, int>::const_iterator it;
    for (it = global_to_local.begin(); it != global_to_local.end(); it++) {
        int global_idx = it->first;
        int local_idx = it->second;
        global_indices[local_idx] = global_idx;
    }
    return global_indices;
}

map<int, vector<double> > construct_data_map(const MatrixD &data)
{
    unsigned int num_rows = data.size1();
    map<int, vector<double> > data_map;
    for (unsigned int row_idx = 0; row_idx < num_rows; row_idx++) {
        data_map[row_idx] = extract_row(data, row_idx);
    }
    return data_map;
}

map<int, int> remove_and_reorder(const map<int, int> &old_global_to_local,
    int global_to_remove)
{
    // extract current ordering
    vector<int> global_indices = extract_global_ordering(old_global_to_local);
    // remove
    int local_to_remove = old_global_to_local.find(global_to_remove)->first;
    global_indices.erase(global_indices.begin() + local_to_remove);
    // constrcut and return
    return construct_lookup_map(global_indices);
}

vector<int> get_indices_to_reorder(const vector<int>
    &data_global_column_indices,
    const map<int, int> &global_to_local)
{
    int num_local_cols = global_to_local.size();
    int num_data_cols = data_global_column_indices.size();
    vector<int> reorder_indices(num_local_cols, -1);
    for (int data_column_idx = 0; data_column_idx < num_data_cols;
        data_column_idx++) {
        int global_column_idx = data_global_column_indices[data_column_idx];
        if (global_to_local.find(global_column_idx) != global_to_local.end()) {
            int local_idx = global_to_local.find(data_column_idx)->second;
            reorder_indices[local_idx] = data_column_idx;
        }
    }
    return reorder_indices;
}

vector<double> reorder_per_indices(const vector<double> &raw_values,
    const vector<int> &reorder_indices)
{
    vector<double> arranged_values;
    vector<int>::const_iterator it;
    for (it = reorder_indices.begin(); it != reorder_indices.end(); it++) {
        int raw_value_idx = *it;
        double raw_value = raw_values[raw_value_idx];
        arranged_values.push_back(raw_value);
    }
    return arranged_values;
}

vector<double> reorder_per_map(const vector<double> &raw_values,
    const vector<int> &global_column_indices,
    const map<int, int> &global_to_local)
{
    vector<int> reorder_indices = \
        get_indices_to_reorder(global_column_indices, global_to_local);
    return reorder_per_indices(raw_values, reorder_indices);
}

vector<vector<double> > reorder_per_map(const vector<vector<double> >
    &raw_values,
    const vector<int> &global_column_indices,
    const map<int, int> &global_to_local)
{
    vector<int> reorder_indices = get_indices_to_reorder(global_column_indices,
            global_to_local);
    vector<vector<double> > arranged_values_v;
    vector<vector<double> >::const_iterator it;
    for (it = raw_values.begin(); it != raw_values.end(); it++) {
        vector<double> arranged_values = reorder_per_indices(*it, reorder_indices);
        arranged_values_v.push_back(arranged_values);
    }
    return arranged_values_v;
}

vector<int> create_sequence(size_t len, int start)
{
    vector<int> sequence(len, 1);
    if (len == 0) {
        return sequence;
    }
    sequence[0] = start;
    std::partial_sum(sequence.begin(), sequence.end(), sequence.begin());
    return sequence;
}

void insert_into_counts(unsigned int draw, vector<int> &counts)
{
    assert(draw <= counts.size());
    if (draw == counts.size()) {
        counts.push_back(1);
    } else {
        counts[draw]++;
    }
}

vector<int> draw_crp_init_counts(int num_datum, double alpha,
    RandomNumberGenerator &rng)
{
    vector<int> counts;
    double rand_u;
    int draw;
    int sum_counts = 0;
    for (int draw_idx = 0; draw_idx < num_datum; draw_idx++) {
        rand_u = rng.next();
        draw = numerics::crp_draw_sample(counts, sum_counts, alpha, rand_u);
        sum_counts++;
        insert_into_counts(draw, counts);
    }
    return counts;
}

vector<vector<int> > draw_crp_init(const vector<int> &global_row_indices,
    double alpha,
    RandomNumberGenerator &rng,
    const string &initialization)
{
    vector<vector<int> > cluster_indices_v;
    if (initialization == TOGETHER) {
        cluster_indices_v.push_back(global_row_indices);
    } else if (initialization == APART) {
        int num_global_row_indices = (int) global_row_indices.size();
        for (int i = 0; i < num_global_row_indices; i++) {
            vector<int> singleton_cluster;
            singleton_cluster.push_back(global_row_indices[i]);
            cluster_indices_v.push_back(singleton_cluster);
        }
    } else if (initialization == FROM_THE_PRIOR) {
        int num_datum = global_row_indices.size();
        vector<int> counts = draw_crp_init_counts(num_datum, alpha, rng);
        vector<int> shuffled_row_indices = global_row_indices;
        random_shuffle(shuffled_row_indices.begin(),
            shuffled_row_indices.end(),
            rng);
        vector<int>::const_iterator it = shuffled_row_indices.begin();
        for (unsigned int cluster_idx = 0; cluster_idx < counts.size();
            cluster_idx++) {
            int count = counts[cluster_idx];
            vector<int> cluster_indices(count, -1);
            std::copy(it, it + count, cluster_indices.begin());
            cluster_indices_v.push_back(cluster_indices);
            it += count;
        }
    } else {
        assert(1 == 0);
        cout << "utils::draw_crp_init: UNKOWN INITIALIZATION: ";
        cout << initialization << endl;
    }
    return cluster_indices_v;
}

vector<vector<vector<int> > > draw_crp_init(const vector<int>
    &global_row_indices,
    const vector<double> &alphas,
    RandomNumberGenerator &rng,
    const string &initialization)
{
    vector<vector<vector<int> > > cluster_indicies_v_v;
    for (vector<double>::const_iterator it = alphas.begin(); it != alphas.end();
        it++) {
        double alpha = *it;
        vector<vector<int> > cluster_indicies_v = draw_crp_init(global_row_indices,
                alpha, rng, initialization);
        cluster_indicies_v_v.push_back(cluster_indicies_v);
    }
    return cluster_indicies_v_v;
}


void copy_column(const MatrixD &fromM, int from_col, MatrixD &toM, int to_col)
{
    size_t row, nrows;
    nrows = fromM.size1();
    assert(nrows == toM.size1());
    for (row = 0; row < nrows; row++) {
        toM(row, to_col) = fromM(row, from_col);
    }
}

MatrixD extract_columns(const MatrixD &fromM, const vector<int> &from_cols)
{
    int num_rows = fromM.size1();
    int num_cols = from_cols.size();
    MatrixD toM(num_rows, num_cols);
    for (int to_col = 0; to_col < num_cols; to_col++) {
        int from_col = from_cols[to_col];
        copy_column(fromM, from_col, toM, to_col);
    }
    return toM;
}

vector<double> extract_columns(const vector<double> &in_vd,
    const vector<int> &from_cols)
{
    vector<double> out_vd;
    vector<int>::const_iterator it;
    for (it = from_cols.begin(); it != from_cols.end(); it++) {
        int from_col = *it;
        out_vd.push_back(in_vd[from_col]);
    }
    return out_vd;
}

int intify(const string &str)
{
    std::istringstream strin(str);
    int str_int;
    strin >> str_int;
    return str_int;
}

vector<double> create_crp_alpha_grid(int n_values, int N_GRID)
{
    vector<double> crp_alpha_grid = log_linspace(1. / n_values, n_values, N_GRID);
    return crp_alpha_grid;
}

void construct_continuous_base_hyper_grids(int n_grid,
    int data_num_vectors,
    vector<double> &r_grid,
    vector<double> &nu_grid)
{
    r_grid = log_linspace(1.0 / data_num_vectors, data_num_vectors, n_grid);
    nu_grid = log_linspace(1.0, data_num_vectors, n_grid);
}

void construct_continuous_specific_hyper_grid(int n_grid,
    const vector<double> &col_data,
    vector<double> &s_grid,
    vector<double> &mu_grid)
{
    // FIXME: should s_grid be a linspace from min el**2 to max el**2
    double sum_sq_deviation, min, max;
    vector<double> filtered_col_data = filter_nans(col_data);
    int num_non_nan = filtered_col_data.size();
    if (num_non_nan != 0) {
        sum_sq_deviation = calc_sum_sq_deviation(filtered_col_data);
        min = *std::min_element(filtered_col_data.begin(), filtered_col_data.end());
        max = *std::max_element(filtered_col_data.begin(), filtered_col_data.end());
    } else {
        // FIXME: What to do here?
        sum_sq_deviation = 100;
        min = -100;
        max = 100;
    }
    s_grid = log_linspace(sum_sq_deviation / 100., sum_sq_deviation, n_grid);
    mu_grid = linspace(min, max, n_grid);
}

void construct_cyclic_base_hyper_grids(int n_grid,
    int data_num_vectors,
    vector<double> &vm_b_grid)
{
    vm_b_grid = linspace(0, 2 * M_PI, n_grid);
}
void construct_cyclic_specific_hyper_grid(int n_grid,
    const vector<double> &col_data,
    vector<double> &vm_a_grid,
    vector<double> &vm_kappa_grid)
{
    double N = (double) col_data.size();
    // double var = calc_sum_sq_deviation(col_data)/N;
    // vm_a_grid = log_linspace(1/var, N/var, n_grid);
    // vm_kappa_grid = log_linspace(var, var*N, n_grid);
    vm_a_grid = log_linspace(1.0 / N, N, n_grid);
    double kappa = numerics::estimate_vonmises_kappa(col_data);
    vm_kappa_grid = linspace(kappa, N * kappa, n_grid);
}

void construct_multinomial_base_hyper_grids(int n_grid,
    int data_num_vectors,
    vector<double> &multinomial_alpha_grid)
{
    multinomial_alpha_grid = log_linspace(1., data_num_vectors, n_grid);
}

int get_vector_num_blocks(
    const vector<int> &vec,
    const map<int, set<int> > &block_lookup)
{

    // Each item is in its own singleton block.
    if (block_lookup.size() == 0) {
        return vec.size();
    }

    int num_items_raw = vec.size();
    int num_items_effective = num_items_raw;

    set<int> seen_items;

    vector<int>::const_iterator it;
    for (it = vec.begin(); it != vec.end(); ++it) {
        int item = *it;
        // Column is not member of a block, continue.
        if (block_lookup.count(item) == 0)
            continue;
        // Column already seen as part of another block, continue.
        if (seen_items.count(item) == 1)
            continue;
        // Item is a member of a block. Find all other members, add them to
        // seen_items, and decrement the effective number of items. Each item is
        // expected to be in its own block, decrement by number of other items
        // (items_in_block.size() - 1).
        set<int> items_in_block = get(block_lookup, item);
        seen_items.insert(items_in_block.begin(), items_in_block.end());
        num_items_effective = num_items_effective - (items_in_block.size() - 1);
    }

    assert(num_items_effective >= 0);
    if (num_items_raw > 0)
        assert(num_items_effective > 0);

    return num_items_effective;
}
