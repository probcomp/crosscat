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
#ifndef GUARD_state_h
#define GUARD_state_h

#include <set>
#include <vector>
#include "View.h"
#include "utils.h"
#include "constants.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>   // for log()
#include <limits>
#include "Matrix.h"

const static std::vector<double> empty_vector_double;

/**
 * A full CrossCat state.  This class is sufficient to draw a posterior sample.
 */
class State
{
public:

    /** Constructor for a fully specified state.
     *  Column and row partitionings are given, as well as all hyper parameters.
     *  \param data The data being modelled
     *  \param GLOBAL_COL_DATATYPES A vector of strings denoting column datatypes.
     *         Valid values are defined in constants.h
     *  \param GLOBAL_COL_MULTINOMIAL_COUNTS A vector of counts, denoting the number
     *         of possible values.
     *  \param global_row_indices A vector of ints, denoting the row indices
     *         of the data matrix passed in
     *  \param global_col_indices A vector of ints, denoting the column indices
     *         of the data matrix passed in
     *  \param HYPERS_M A map of column index to column hypers
     *  \param column_partition The partitioning of column indices.  Each partition
     *         denoting a view
     *  \param col_ensure_dep
     *  \param col_ensure_ind
     *  \param COLUMN_CRP_ALPHA The column CRP hyperparameter
     *  \param row_partition_v A vector of row partitionings.  One row partitioning
     *         for each element of column_partition
     *  \param row_crp_alpha_v The row CRP hyperparameters.  One for each element of
     *         column_partition
     *  \param N_GRID The number of grid points to use when gibbs sampling hyperparameters
     *  \param SEED The seed for the state's RNG
     */
    State(const MatrixD &data,
        const std::vector<std::string> &GLOBAL_COL_DATATYPES,
        const std::vector<int> &GLOBAL_COL_MULTINOMIAL_COUNTS,
        const std::vector<int> &global_row_indices,
        const std::vector<int> &global_col_indices,
        const std::map<int, CM_Hypers> &HYPERS_M,
        const std::vector<std::vector<int> > &column_partition,
        const std::map<int, std::set<int> > &col_ensure_dep,
        const std::map<int, std::set<int> > &col_ensure_ind,
        double COLUMN_CRP_ALPHA,
        const std::vector<std::vector<std::vector<int> > > &row_partition_v,
        const std::vector<double> &row_crp_alpha_v,
        const std::vector<double> &ROW_CRP_ALPHA_GRID = empty_vector_double,
        const std::vector<double> &COLUMN_CRP_ALPHA_GRID = empty_vector_double,
        const std::vector<double> &S_GRID = empty_vector_double,
        const std::vector<double> &MU_GRID = empty_vector_double,
        int N_GRID = 31, int SEED = 0, int CT_KERNEL = 0);

    /** Constructor for drawing a CrossCat state from the prior.
     *  Column and row partitionings are given, as well as all hyper parameters.
     *  \param data The data being modelled
     *  \param GLOBAL_COL_DATATYPES A vector of strings denoting column datatypes.
     *         Valid values are defined in constants.h
     *  \param GLOBAL_COL_MULTINOMIAL_COUNTS A vector of counts, denoting the number
     *         of possible values.
     *  \param col_ensure_dep
     *  \param col_ensure_ind
     *  \param global_row_indices A vector of ints, denoting the row indices
     *         of the data matrix passed in
     *  \param global_col_indices A vector of ints, denoting the column indices
     *         of the data matrix passed in
     *  \param col_initialization A string denoting which type of intialization
     *         to use for the column partitioning.  Valid values are defined in constants.h
     *  \param row_initialization A tring denoting which type of initialization
     *         to use for the row partitioning.  Valid values are defined in constants.h
     *  \param N_GRID The number of grid points to use when gibbs sampling hyperparameters
     *  \param SEED The seed for the state's RNG
     */
    State(const MatrixD &data,
        const std::vector<std::string> &GLOBAL_COL_DATATYPES,
        const std::vector<int> &GLOBAL_COL_MULTINOMIAL_COUNTS,
        const std::vector<int> &global_row_indices,
        const std::vector<int> &global_col_indices,
        const std::string &col_initialization = FROM_THE_PRIOR,
        std::string row_initialization = "",
        const std::vector<double> &ROW_CRP_ALPHA_GRID = empty_vector_double,
        const std::vector<double> &COLUMN_CRP_ALPHA_GRID = empty_vector_double,
        const std::vector<double> &S_GRID = empty_vector_double,
        const std::vector<double> &MU_GRID = empty_vector_double,
        int N_GRID = 31, int SEED = 0, int CT_KERNEL = 0);

    ~State();

    //
    // getters
    //
    std::map<int, std::set<int> > get_column_dependencies() const;
    std::map<int, std::set<int> > get_column_independencies() const;
    std::vector<int> get_column_dependencies(int feature_idx) const;
    std::vector<int> get_column_independencies(int feature_idx) const;
    /**
     * \return The number of columns in the state
     */
    int get_num_cols() const;
    /**
     * \return The number of effective columns in the state, treating each block
     * of dependent columns as a single column
     */
    int get_num_cols_effective() const;
    /**
     * \return The number of views (column partitions)
     */
    int get_num_views() const;
    /**
     * \return The number of columns in each view
     */
    std::vector<int> get_view_counts() const;
    /**
     * \return the column partition CRP hyperparameter
     */
    double get_column_crp_alpha() const;
    /**
     * \return The contribution of the column CRP marginal log probability
     * to the state's marginal log probability
     */
    double get_column_crp_score() const;
    /**
     * \return The contribution of each View's row clustering marginal log probability
     * to the state's marginal log probability
     */
    double get_data_score() const;
    /**
     * \return The state's marginal log probability
     */
    double get_marginal_logp() const;
    /**
     * \return The column indices in each column partition
     */
    std::map<int, std::vector<int> > get_column_groups() const;
    /**
     * \return A uniform random draw from [0, 1] using the state's rng
     */
    double draw_rand_u();
    /**
     * \return A random int from [0, max] using the state's rng
     */
    int draw_rand_i(int max = MAX_INT);

    //
    // helpers for API
    //
    /**
     * Get the hyperparameters used for the ith view
     * \return A map from hyperparameter name to value
     */
    std::map<std::string, double> get_row_partition_model_hypers_i(
        int view_idx) const;
    /**
     * Get the row partition model counts for the ith view
     * \return a vector of ints
     */
    std::vector<int> get_row_partition_model_counts_i(int view_idx) const;
    /**
     * Get the sufficient statistics for the ith view
     * \return A vector of cluster sufficient statistics
     */
    std::vector<std::vector<std::map<std::string, double> > >
    get_column_component_suffstats_i(int view_idx) const;
    /**
     * Get all the column component model hyperparameters in order
     */
    std::vector<CM_Hypers> get_column_hypers() const;
    /**
     * Get the hyperparameter associated with the column CRP model
     */
    std::map<std::string, double> get_column_partition_hypers() const;
    /**
     * Get a list denoting which view each column belongs to
     */
    std::vector<int> get_column_partition_assignments() const;
    /**
     * Get a list of counts of columns in each view
     */
    std::vector<int> get_column_partition_counts() const;
    /**
     * Get a list of cluster memberships for each view.
     * Each cluster membership is itself a list denoting which cluster a row belongs to
     */
    std::vector<std::vector<int> > get_X_D() const;
    /**
     * Draw a sample row based on an existing row
     */
    std::vector<double> get_draw(int row_idx, int random_seed) const;

    double insert_row(const std::vector<double> &row_data, int matching_row_idx,
        int row_idx = -1);
    //
    // mutators
    //
    /**
     * Insert feature_data into the view specified by which_view.  feature_idx
     * is the column index to associate with it
     * \param feature_idx The column index that the view should associate with the data
     * \param feature_data The data that comprises the feature
     * \param which_view A reference to the view in which the feature should be added
     * \return The delta in the state's marginal log probability
     */
    double insert_feature(int feature_idx,
        const std::vector<double> &feature_data,
        View &which_view);
    /**
     * Gibbs sample which view to insert the feature into.
     * \param feature_idx The column index that the view should associate with the data
     * \param feature_data The data that comprises the feature
     * \param singleton_view A reference to an empty view to allow for creation of new views.
     *        Deleted internally if not used.
     */
    double sample_insert_feature(int feature_idx,
        const std::vector<double> &feature_data,
        View &singleton_view);
    /**
     * Gibbs sample which view to insert the feature block into.
     * \param feature_idxs The column indexes that the view should associate with the data.
     * \param feature_datas The vector of data that comprises the features.
     * \param singleton_view A reference to an empty view to allow for creation of new views.
     *        Deleted internally if not used.
     */
    double sample_insert_feature_block(
        const std::vector<int> &feature_idxs,
        const std::vector<std::vector<double> > &feature_datas,
        View &singleton_view);
    /**
     * Remove a feature from the state.
     * \param feature_idx The column index that the view should associaate with the data
     * \param feature_data The data that comprises the feature
     * \param p_singleton_view A pointer to the view the feature was removed from.
     *        This variables name is a bit of a misnomer: its not necessarily a singleton.
     *        Necesary to pass out for determining the marginal log probability delta
     */
    double remove_feature(int feature_idx,
        const std::vector<double> &feature_data,
        View *&p_singleton_view);
    /**
     * Remove a feature from the state.
     * \param feature_idx The column index that the view should associaate with the data
     * \param feature_data The data that comprises the feature
     */
    double remove_feature(
        int feature_idx,
        const std::vector<double> &feature_data);
    /**
     * Gibbs sample a feature among the views, possibly creating a new view
     * \param feature_idx The column index that the view should associaate with the data
     * \param feature_data The data that comprises the feature
     */
    double transition_feature_gibbs(int feature_idx,
        const std::vector<double> &feature_data);
    /**
     * Gibbs sample a block of dependent features among the views, possibly creating a new view
     * \param feature_idxs The column indexes that the view should associate with the data
     * \param feature_datas The vector of data that comprises the features
     */
    double transition_feature_block_gibbs(
        const std::vector<int> &feature_idxs,
        const std::vector<std::vector<double> > &feature_datas);
    /**
     * Helper for transition_feature_mh
     * \param feature_idx The column index that the view should associaate with the data
     * \param feature_data The data that comprises the feature
     * \param proposed_view The view to propose jumping to
     */
    double mh_choose(int feature_idx,
        const std::vector<double> &feature_data,
        View &proposed_view);
    double get_proposal_logp(View &proposed_view);
    double get_proposal_log_ratio(View &from_view, View &to_view);
    /**
     * Metropolis birth-death process for assigning columns to view (or creating new views)
     * \param feature_idx The column index that the view should associaate with the data
     * \param feature_data The data that comprises the feature
     */
    double transition_feature_mh(int feature_idx,
        const std::vector<double> &feature_data);
    /**
     * Instantiate a new view object with properties matching the state
     * (datatypes, #rows, etc) and track in memeber variable views
     */
    View &get_new_view();
    /**
     * Get a particular view.
     */
    View &get_view(int view_idx);
    /**
     * Deallocate and remove the state if its empty.  Used as a helper for feature transitions
     */
    void remove_if_empty(View &which_view);
    /**
     * Deallocate all data structures.  For use before exiting.
     */
    void remove_all();
    /**
     * Stale function: don't use
     */
    double transition_view_i(int which_view,
        const std::map<int, std::vector<double> > &row_data_map);
    /**
     * Stale function: don't use
     */
    double transition_view_i(int which_view, const MatrixD &data);
    /**
     * Stale function: don't use
     */
    double transition_views(const MatrixD &data);
    /**
     * Stale function: don't use
     */
    double transition_views_row_partition_hyper();
    /**
     * Stale function: don't use
     */
    double transition_views_col_hypers();
    /**
     * Stale function: don't use
     */
    double transition_views_zs(const MatrixD &data);
    /**
     * Stale function: don't use
     */
    double transition(const MatrixD &data);
    //
    /**
     * Gibbs sample column CRP hyperparameter over its hyper grid
     * \return The delta in the state's marginal log probability
     */
    double transition_column_crp_alpha();
    /**
     * Gibbs sample view memebership of specified feature (column) indices
     * \return The delta in the state's marginal log probability
     */
    double transition_features(const MatrixD &data,
        std::vector<int> which_features);
    /**
     * Gibbs sample component model hyperparameters of specified feature (column) indices
     * \return The delta in the state's marginal log probability
     */
    double transition_column_hyperparameters(std::vector<int> which_cols);
    /**
     * Gibbs sample row partition CRP hyperparameter on views denoted by specified column indices
     * \return The delta in the state's marginal log probability
     */
    double transition_row_partition_hyperparameters(const std::vector<int>
        &which_cols);
    /**
     * Gibbs sample cluster membership of specified rows
     * \return The delta in the state's marginal log probability
     */
    double transition_row_partition_assignments(const MatrixD &data,
        std::vector<int> which_rows);
    //
    // calculators
    /**
     * \return The crp probability of a feature belonging to a particular view
     */
    double calc_feature_view_crp_logp(
        const View &v,
        const int &global_col_idx) const;
    /**
     * \return The crp probabilities of a feature belonging to each view.
     */
    std::vector<double> calc_feature_view_crp_logps(
        const int &global_col_idx) const;
    /**
     * \return The probability of feature data under row partition of a particular view
     */
    double calc_feature_view_data_logp(
        const std::vector<double> &col_data,
        const std::string &col_datatype,
        const View &v,
        const CM_Hypers &hypers,
        const int &global_col_idx) const;
    /**
     * \return The probability of feature data under row partition of each view.
     */
    std::vector<double> calc_feature_view_data_logps(
        const std::vector<double> &col_data,
        const int &global_col_idx) const;
    /**
     * \return The predictive log likelihood of a feature belonging to a particular view
     */
    double calc_feature_view_predictive_logp(
        const std::vector<double> &col_data,
        const std::string &col_datatype,
        const View &v,
        double &crp_log_delta,
        double &data_log_delta,
        const CM_Hypers &hypers,
        const int &global_col_idx) const;
    /**
     * \return The predictive log likelihoods of a feature belonging to each view
     */
    std::vector<double> calc_feature_view_predictive_logps(
        const std::vector<double> &col_data,
        int global_col_idx) const;
    /**
     * \return The predictive log likelihoods of a feature block belonging to each view.
     */
    std::vector<double> calc_feature_view_predictive_logps_block(
        const std::vector<int> &feature_idxs,
        const std::vector<std::vector<double> > &feature_datas) const;
    /**
     * \return The predictive log likelihood of a row having been generated by this state
     */
    double calc_row_predictive_logp(const std::vector<double> &in_vd);
    //
    // helpers
    /**
     * \return true if view contains a column which is independent of global_col_idx.
     */
    bool view_violates_independency(
        const View &view, const int &global_col_idx) const;
    /**
     * \return The log likelihood of the column CRP hyperparmeter value
     * given the state's column partitioning and the hyperprior on alpha
     * defined in numerics::calc_crp_alpha_hyperprior
     */
    double calc_column_crp_marginal() const;
    /**
     * \return The log likelihoods of the given column CRP hyperparmeter values
     * given the state's column partitioning and the hyperprior on alpha
     * defined in numerics::calc_crp_alpha_hyperprior
     */
    std::vector<double> calc_column_crp_marginals(const std::vector<double> &
        alphas_to_score) const;
    friend std::ostream &operator<<(std::ostream &os, const State &s);
    std::string to_string(const std::string &join_str = "\n",
        bool top_level = false) const;
private:
    DISALLOW_COPY_AND_ASSIGN(State);
    // parameters
    std::map<int, std::string> global_col_datatypes;
    std::map<int, int> global_col_multinomial_counts;
    std::map<int, CM_Hypers> hypers_m;
    double column_crp_alpha;
    double column_crp_score;
    double data_score;
    int ct_kernel;
    // column structure ensure
    std::map<int, std::set<int> > column_dependencies;
    std::map<int, std::set<int> > column_independencies;
    int num_cols_effective;
    // grids
    std::vector<double> column_crp_alpha_grid;
    std::vector<double> row_crp_alpha_grid;
    std::vector<double> r_grid;
    std::vector<double> nu_grid;
    std::vector<double> vm_b_grid;
    std::vector<double> multinomial_alpha_grid;
    std::map<int, std::vector<double> > s_grids;
    std::map<int, std::vector<double> > mu_grids;
    std::map<int, std::vector<double> > vm_a_grids;
    std::map<int, std::vector<double> > vm_kappa_grids;
    // lookups
    std::vector<View *> views;
    std::map<int, View *> view_lookup; // global_column_index to View mapping
    // sub-objects
    RandomNumberGenerator rng;
    // resources
    void increment_num_cols_effective();
    void decrement_num_cols_effective();
    void construct_base_hyper_grids(const matrix<double> &
        data, int N_GRID,
        std::vector<double> ROW_CRP_ALPHA_GRID,
        std::vector<double> COLUMN_CRP_ALPHA_GRID);
    void construct_column_hyper_grids(const matrix<double> &
        data,
        const std::vector<int> &global_col_indices,
        const std::vector<std::string> &global_col_datatypes,
        const std::vector<double> &S_GRID,
        const std::vector<double> &MU_GRID);
    CM_Hypers get_default_hypers() const;
    double sample_column_crp_alpha();
    double sample_row_crp_alpha();
    std::vector<double> sample_row_crp_alphas(int N_views);
    std::vector<std::vector<int> > generate_col_partition(const std::vector<int> &
        global_col_indices,
        const std::string &col_initialization);
    std::vector<std::vector<std::vector<int> > > generate_row_partitions(
        const std::vector<int> &global_row_indices,
        const std::vector<double> &row_crp_alpha_v,
        const std::string &row_initialization);
    void init_base_hypers();
    CM_Hypers uniform_sample_hypers(int global_col_idx);
    void init_column_hypers(const std::vector<int> &global_col_indices);
    void init_views(const MatrixD &data,
        const std::vector<int> &global_row_indices,
        const std::vector<int> &global_col_indices,
        const std::vector<std::vector<int> > &column_partition,
        const std::vector<std::vector<std::vector<int> > > &row_partition_v,
        const std::vector<double> &row_crp_alpha_v);
};

#endif // GUARD_state_h
