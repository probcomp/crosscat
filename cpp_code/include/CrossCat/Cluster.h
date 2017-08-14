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
#ifndef GUARD_cluster_h
#define GUARD_cluster_h

// #include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <cstdio>
#include <cassert>
//
#include "utils.h"
#include "constants.h"
#include "ComponentModel.h"
#include "ContinuousComponentModel.h"
#include "CyclicComponentModel.h"
#include "MultinomialComponentModel.h"


class Cluster
{
public:
    Cluster(const std::vector<CM_Hypers *> &hypers_v);
    Cluster();
    //
    // getters
    int get_num_cols() const;
    int get_count() const;
    double get_marginal_logp() const;
    std::map<std::string, double> get_suffstats_i(int idx) const;
    CM_Hypers get_hypers_i(int idx) const;
    std::set<int> get_row_indices_set() const;
    std::vector<int> get_row_indices_vector() const;
    std::vector<double> get_draw(int random_seed) const;
    //
    // calculators
    std::vector<double> calc_marginal_logps() const;
    double calc_sum_marginal_logps() const ;
    double calc_row_predictive_logp(const std::vector<double> &vd) const;
    std::vector<double> calc_hyper_conditionals(int which_col,
        const std::string &which_hyper,
        const std::vector<double> &hyper_grid) const;
    double calc_column_predictive_logp(const std::vector<double> &column_data,
        const std::string &col_datatype,
        const std::vector<int> &data_global_row_indices,
        const CM_Hypers &hypers);
    //
    // mutators
    double insert_row(const std::vector<double> &values, int row_idx);
    double remove_row(const std::vector<double> &values, int row_idx);
    double remove_col(int col_idx);
    double insert_col(const std::vector<double> &data,
        const std::string &col_datatype,
        const std::vector<int> &data_global_row_indices,
        const CM_Hypers &hypers);
    double incorporate_hyper_update(int which_col);
    void delete_component_models(bool check_empty = true);
    //
    // helpers
    friend std::ostream &operator<<(std::ostream &os, const Cluster &c);
    std::string to_string(const std::string &join_str = "\n",
        bool top_level = false) const;
    //
    // make private later
    std::vector<ComponentModel *> p_model_v;
private:
    double score;
    void init_columns(const std::vector<CM_Hypers *> &hypers_v);
    std::set<int> row_indices;
};

#endif // GUARD_cluster_h
