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
#ifndef GUARD_view_h
#define GUARD_view_h


#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <numeric> // std::accumulate
//
#include "RandomNumberGenerator.h"
#include "utils.h"
#include "Cluster.h"
#include "Matrix.h"
#include "numerics.h"

class Cluster;

class View
{
public:
    //FIXME: add constructor with ranges as arguments, rather than recalculate
    View(const MatrixD &data,
        const std::map<int, std::string> &GLOBAL_COL_DATATYPES,
        const std::vector<std::vector<int> > &row_partitioning,
        const std::vector<int> &global_row_indices,
        const std::vector<int> &global_col_indices,
        const int &num_cols_effective,
        std::map<int, CM_Hypers> &hypers_m,
        const std::vector<double> &ROW_CRP_ALPHA_GRID,
        const std::vector<double> &MULTINOMIAL_ALPHA_GRID,
        const std::vector<double> &R_GRID,
        const std::vector<double> &NU_GRID,
        const std::vector<double> &VM_B_GRID,
        const std::map<int, std::vector<double> > &S_GRIDS,
        const std::map<int, std::vector<double> > &MU_GRIDS,
        const std::map<int, std::vector<double> > &VM_A_GRIDS,
        const std::map<int, std::vector<double> > &VM_KAPPA_GRIDS,
        double CRP_ALPHA,
        int SEED = 0);
    View(const MatrixD &data,
        const std::map<int, std::string> &GLOBAL_COL_DATATYPES,
        const std::vector<int> &global_row_indices,
        const std::vector<int> &global_col_indices,
        std::map<int, CM_Hypers> &hypers_m,
        const std::vector<double> &ROW_CRP_ALPHA_GRID,
        const std::vector<double> &MULTINOMIAL_ALPHA_GRID,
        const std::vector<double> &R_GRID,
        const std::vector<double> &NU_GRID,
        const std::vector<double> &VM_B_GRID,
        const std::map<int, std::vector<double> > &S_GRIDS,
        const std::map<int, std::vector<double> > &MU_GRIDS,
        const std::map<int, std::vector<double> > &VM_A_GRIDS,
        const std::map<int, std::vector<double> > &VM_KAPPA_GRIDS,
        int SEED = 0);
    View(const std::map<int, std::string> &GLOBAL_COL_DATATYPES,
        const std::vector<int> &global_row_indices,
        const std::vector<double> &ROW_CRP_ALPHA_GRID,
        const std::vector<double> &MULTINOMIAL_ALPHA_GRID,
        const std::vector<double> &R_GRID,
        const std::vector<double> &NU_GRID,
        const std::vector<double> &VM_B_GRID,
        const std::map<int, std::vector<double> > &S_GRIDS,
        const std::map<int, std::vector<double> > &MU_GRIDS,
        const std::map<int, std::vector<double> > &VM_A_GRIDS,
        const std::map<int, std::vector<double> > &VM_KAPPA_GRIDS,
        int SEED = 0);
    //
    // getters (external use)
    double get_num_vectors() const;
    double get_num_cols() const;
    int get_num_cols_effective() const;
    int get_num_clusters() const;
    double get_crp_score() const;
    double get_data_score() const;
    double get_score() const;
    double get_crp_alpha() const;
    std::vector<double> get_crp_alpha_grid() const;
    std::vector<std::string> get_hyper_strings(int which_col);
    std::vector<double> get_hyper_grid(int global_col_idx,
        const std::string &which_hyper);
    CM_Hypers get_hypers(int local_col_idx) const;
    //
    // API helpers
    std::map<std::string, double> get_row_partition_model_hypers() const;
    std::vector<int> get_row_partition_model_counts() const;
    std::vector<std::map<std::string, double> > get_column_component_suffstats_i(
        int global_col_idx) const;
    std::vector<std::vector<std::map<std::string, double> > > \
    get_column_component_suffstats() const;
    std::vector<int> get_global_col_indices();
    std::vector<double> get_draw(int row_idx, int random_seed) const;
    //
    // getters (internal use)
    Cluster &get_cluster(int cluster_idx);
    std::vector<int> get_cluster_counts() const;
    //
    // calculators
    double calc_cluster_vector_predictive_logp(const std::vector<double> &vd,
        const Cluster &cd,
        double &crp_logp_delta,
        double &data_logp_delta) const;
    std::vector<double> calc_cluster_vector_predictive_logps(
        const std::vector<double> &vd);
    double calc_crp_marginal() const;
    std::vector<double> calc_crp_marginals(const std::vector<double> &alphas) const;
    std::vector<double> calc_hyper_conditionals(int which_col,
        const std::string &which_hyper,
        const std::vector<double> &hyper_grid) const;
    double calc_column_predictive_logp(const std::vector<double> &column_data,
        const std::string &col_datatype,
        const std::vector<int> &data_global_row_indices,
        const CM_Hypers &hypers) const;
    //
    // mutators
    void set_row_partitioning(const std::vector<std::vector<int> >
        &row_partitioning);
    void set_row_partitioning(const std::vector<int> &global_row_indices);
    double set_crp_alpha(double new_crp_alpha);
    Cluster &get_new_cluster();
    double insert_row(const std::vector<double> &vd, Cluster &cd, int row_idx);
    double insert_row(const std::vector<double> &vd, int matching_row_idx,
        int row_idx);
    double insert_row(const std::vector<double> &vd, int row_idx);
    double remove_row(const std::vector<double> &vd, int row_idx);
    double remove_col(int global_col_idx);
    double insert_col(const std::vector<double> &col_data,
        const std::vector<int> &data_global_row_indices,
        int global_col_idx,
        CM_Hypers &hypers);
    double insert_cols(const MatrixD &data,
        const std::vector<int> &global_row_indices,
        const std::vector<int> &global_col_indices,
        std::map<int, CM_Hypers> &hypers_m);
    void remove_if_empty(Cluster &which_cluster);
    void remove_all();
    double transition_z(const std::vector<double> &vd, int row_idx);
    double transition_zs(const std::map<int, std::vector<double> > &row_data_map);
    double transition_crp_alpha();
    double set_hyper(int which_col, const std::string &which_hyper,
        double new_value);
    double transition_hyper_i(int which_col, const std::string &which_hyper,
        const std::vector<double> &hyper_grid);
    double transition_hyper_i(int which_col, const std::string &which_hyper);
    double transition_hypers_i(int which_col);
    double transition_hypers();
    double transition(const std::map<int, std::vector<double> > &row_data_map);
    void increment_num_cols_effective();
    void decrement_num_cols_effective();
    //
    // data structures
    std::vector<Cluster *> clusters;
    std::map<int, Cluster *> cluster_lookup;
    std::vector<CM_Hypers *> hypers_v;
    //
    // helper functions
    std::vector<double> align_data(const std::vector<double> &values,
        const std::vector<int> &global_column_indices) const;
    std::vector<int> shuffle_row_indices();
    std::vector<std::vector<int> > get_cluster_groupings() const;
    std::vector<int> get_canonical_clustering() const;
    //
    friend std::ostream &operator<<(std::ostream &os, const View &v);
    std::string to_string(const std::string &join_str = "\n",
        bool top_level = false) const;
    void print();
    void print_score_matrix();
    // void assert_state_consistency();
    // double score_test_set(const std::vector<std::vector<double> >& test_set) const;
    //
    // hyper inference grids FIXME: MOVE TO PRIVATE WHEN DONE TESTING
    std::map<int, int> global_to_local; // FIXME: specify appicability to columns
private:
    // parameters
    double crp_alpha;
    double crp_score;
    double data_score;
    int num_cols_effective;
    std::map<int, std::string> global_col_datatypes;
    //  grids
    std::vector<double> crp_alpha_grid;
    std::vector<double> multinomial_alpha_grid;
    std::vector<double> r_grid;
    std::vector<double> nu_grid;
    std::vector<double> vm_b_grid;
    std::map<int, std::vector<double> > s_grids;
    std::map<int, std::vector<double> > mu_grids;
    std::map<int, std::vector<double> > vm_a_grids;
    std::map<int, std::vector<double> > vm_kappa_grids;
    // sub-objects
    RandomNumberGenerator rng;
    // resources
    double draw_rand_u();
    int draw_rand_i(int max);
    // helpers
    void construct_base_hyper_grids(int num_rows);
    void construct_column_hyper_grid(const std::vector<double> &col_data,
        int gobal_col_idx);
    /* CM_Hypers data_hypers; */
};

#endif //GUARD_view_h
