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
#ifndef GUARD_utils_h
#define GUARD_utils_h

#include "numerics.h"
#include "constants.h"
#include "RandomNumberGenerator.h"
#include "Matrix.h"
//
#include <iostream>
#include <string>
#include <sstream> // stringstream in stringify()
#include <set>
#include <map>
#include <cmath> // isnan, isfinite

typedef std::map<std::string, double> ComponentModelHypers;
typedef ComponentModelHypers CM_Hypers;

template <class K, class V>
std::ostream &operator<<(std::ostream &os, const std::map<K, V> &in_map)
{
    os << "{";
    typename std::map<K, V>::const_iterator it = in_map.begin();
    if (it != in_map.end()) {
        os << it->first << ":" << it->second;
        ++it;
    }
    for (; it != in_map.end(); ++it) {
        os << ", " << it->first << " : " << it->second;
    }
    os << "}";
    return os;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const std::set<T> &sT)
{
    os << "{";
    typename std::set<T>::const_iterator it = sT.begin();
    if (it != sT.end()) {
        os << *it;
        ++it;
    }
    for (; it != sT.end(); ++it) {
        os << ", " << *it;
    }
    os << "}";
    return os;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vT)
{
    os << "[";
    typename std::vector<T>::const_iterator it = vT.begin();
    if (it != vT.end()) {
        os << *it;
        ++it;
    }
    for (; it != vT.end(); ++it) {
        os << ", " << *it;
    }
    os << "]";
    return os;
}

bool is_almost(double val1, double val2, double precision);

std::vector<double> linspace(double a, double b, size_t n);
std::vector<double> log_linspace(double a, double b, size_t n);
std::vector<int> create_sequence(size_t len, int start = 0);

std::vector<double> std_vector_divide_elemwise(
    const std::vector<double> &vec, const double &val);

std::vector<double> std_vector_add(const std::vector<double> &vec1,
    const std::vector<double> &vec2);
std::vector<double> std_vector_add(const std::vector<std::vector<double> >
    &vec_vec);

double std_vector_sum(const std::vector<double> &vec);
double std_vector_mean(const std::vector<double> &vec);
double calc_sum_sq_deviation(const std::vector<double> &values);
std::vector<double> extract_row(const MatrixD &data, int row_idx);
std::vector<double> extract_col(const MatrixD &data, int col_idx);
std::vector<std::vector<double> > extract_cols(
    const MatrixD &data, std::vector<int> &col_idxs);

template <class T>
std::vector<T> append(const std::vector<T> &vec1, const std::vector<T> &vec2)
{
    std::vector<T> vec = vec1;
    vec.insert(vec.end(), vec2.begin(), vec2.end());
    return vec;
}

template <class K, class V>
V setdefault(std::map<K, V> &m, const K &key, const V &value)
{
    typename std::map<K, V>::const_iterator it = m.find(key);
    if (it == m.end()) {
        return value;
    } else {
        return it->second;
    }
}

template <class K, class V>
K get_key_of_value(const std::map<K, V> &m, const V &value)
{
    typename std::map<K, V>::const_iterator it;
    for (it = m.begin(); it != m.end(); ++it) {
        if (it->second == value) {
            break;
        }
    }
    assert(it != m.end());
    return it->first;
}

template <class K, class V>
V get(const std::map<K, V> &m, const K &key)
{
    typename std::map<K, V>::const_iterator it = m.find(key);
    assert(it != m.end());
    return it->second;
}

std::vector<int> extract_global_ordering(const std::map<int, int>
    &global_to_local);

template <class K, class V>
std::map<K, V> construct_lookup_map(const std::vector<K> &keys,
    const std::vector<V> &values)
{
    assert(keys.size() == values.size());
    std::map<K, V> lookup;
    for (unsigned int idx = 0; idx < keys.size(); idx++) {
        lookup[keys[idx]] = values[idx];
    }
    return lookup;
}

template <class K>
std::map<K, int> construct_lookup_map(const std::vector<K> &keys)
{
    return construct_lookup_map(keys, create_sequence(keys.size()));
}

std::map<int, std::vector<double> > construct_data_map(const MatrixD &data);
std::map<int, int> remove_and_reorder(const std::map<int, int> &global_to_local,
    int global_to_remove);

std::vector<int> get_indices_to_reorder(const std::vector<int> &
    data_global_column_indices,
    const std::map<int, int> &global_to_local);
std::vector<double> reorder_per_indices(const std::vector<double> &raw_values,
    const std::vector<int> &reorder_indices);
std::vector<double> reorder_per_map(const std::vector<double> &raw_values,
    const std::vector<int> &global_column_indices,
    const std::map<int, int> &global_to_local);
std::vector<std::vector<double> > reorder_per_map(
    const std::vector<std::vector<double> > &raw_values,
    const std::vector<int> &global_column_indices,
    const std::map<int, int> &global_to_local);

std::vector<std::vector<int> > draw_crp_init(
    const std::vector<int> &global_row_indices,
    double alpha,
    RandomNumberGenerator &rng,
    const std::string &initialization = FROM_THE_PRIOR);

std::vector<std::vector<std::vector<int> > > draw_crp_init(
    const std::vector<int> &global_row_indices,
    const std::vector<double> &alphas,
    RandomNumberGenerator &rng,
    const std::string &initialization);

void copy_column(const MatrixD &fromM, int from_col, MatrixD &toM, int to_col);
MatrixD extract_columns(const MatrixD &fromM,
    const std::vector<int> &from_cols);
std::vector<double> extract_columns(const std::vector<double> &in_vd,
    const std::vector<int> &from_cols);

template <class T>
std::vector<T> set_to_vector(const std::set<T> &in_set)
{
    std::vector<T> out_vector;
    typename std::set<T>::const_iterator it;
    for (it = in_set.begin(); it != in_set.end(); ++it) {
        T element = *it;
        out_vector.push_back(element);
    }
    return out_vector;
}


template <class T>
std::map<T, int> set_to_map(const std::set<T> &in_set)
{
    std::map<T, int> out_map;
    typename std::set<T>::const_iterator it;
    for (it = in_set.begin(); it != in_set.end(); ++it) {
        T element = *it;
        int out_map_size = out_map.size();
        out_map[element] = out_map_size;
    }
    return out_map;
}

template <class T>
std::set<T> array_to_set(size_t num_items, T *arr) {
    std::set<T> out_set;
    for (size_t i = 0; i < num_items; i++){
        T next_item = *arr;
        out_set.insert(next_item);
        arr++;
    }
    return out_set;
}


template <class T>
std::map<T, int> vector_to_map(const std::vector<T> &in_vector)
{
    std::map<T, int> out_map;
    typename std::vector<T>::const_iterator it;
    for (it = in_vector.begin(); it != in_vector.end(); ++it) {
        T element = *it;
        int out_map_size = out_map.size();
        out_map[element] = out_map_size;
    }
    return out_map;
}

template <class T>
std::string stringify(const T &element)
{
    std::stringstream ss;
    ss << element;
    return ss.str();
}

int intify(const std::string &str);

template <class K, class V>
std::map<V, std::set<K> > group_by_value(const std::map<K, V> &in_map)
{
    std::map<V, std::set<K> > out_map;
    typename std::map<K, V>::const_iterator it;
    for (it = in_map.begin(); it != in_map.end(); ++it) {
        K k = it->first;
        V v = it->second;
        out_map[v].insert(k);
    }
    return out_map;
}

template <class V>
std::vector<int> define_group_ordering(const std::map<int, V> &local_lookup,
    const std::vector<V> &in_vector)
{
    std::vector<int> group_ordering;
    std::map<V, int> V_to_int = vector_to_map(in_vector);
    int num_elements = local_lookup.size();
    for (int element_idx = 0; element_idx < num_elements; element_idx++) {
        V v = get(local_lookup, element_idx);
        int group_idx = V_to_int[v];
        group_ordering.push_back(group_idx);
    }
    return group_ordering;
}

// semi numeric functions
std::vector<double> create_crp_alpha_grid(int n_values, int N_GRID);
void construct_continuous_base_hyper_grids(int n_grid,
    int data_num_vectors,
    std::vector<double> &r_grid,
    std::vector<double> &nu_grid);
void construct_continuous_specific_hyper_grid(int n_grid,
    const std::vector<double> &col_data,
    std::vector<double> &s_grid,
    std::vector<double> &mu_grid);

void construct_cyclic_base_hyper_grids(int n_grid,
    int data_num_vectors,
    std::vector<double> &vm_b_grid);
void construct_cyclic_specific_hyper_grid(int n_grid,
    const std::vector<double> &col_data,
    std::vector<double> &vm_a_grid,
    std::vector<double> &vm_kappa);

void construct_multinomial_base_hyper_grids(int n_grid,
    int data_num_vectors,
    std::vector<double> &multinomial_alpha_grid);

// See test_utils.test_get_vector_num_blocks for explanation.
int get_vector_num_blocks(
    const std::vector<int> &vec,
    const std::map<int, std::set<int> > &block_lookup);

template <class T>
matrix<T> vector_to_matrix(const std::vector<T> &vT)
{
    matrix<T> matrix_out(1, vT.size());
    for (unsigned int i = 0; i < vT.size(); i++) {
        matrix_out(0, i) = vT[i];
    }
    return matrix_out;
}

template <class T>
int count_elements(const std::vector<std::vector<T> > &v_v_T)
{
    int num_elements = 0;
    typename std::vector<std::vector<T> >::const_iterator it;
    for (it = v_v_T.begin(); it != v_v_T.end(); ++it) {
        num_elements += (*it).size();
    }
    return num_elements;
}

#define DISALLOW_COPY_AND_ASSIGN(CLASS)         \
  CLASS(const CLASS&);                          \
  void operator=(const CLASS&)

template <class T>
void
random_shuffle(T begin, T end, RandomNumberGenerator &rng)
{
    typename std::iterator_traits<T>::difference_type n = end - begin;
    for (int i = 0; i < n; i++) {
        std::swap(begin[i], begin[rng.nexti(i + 1)]);
    }
}

#endif // GUARD_utils_H
