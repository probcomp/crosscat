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

#include "Matrix.h"

#include "View.h"

using namespace std;

// row partitioning, row_crp_alpha fully specified
View::View(const MatrixD &data,
    const map<int, string> &GLOBAL_COL_DATATYPES,
    const vector<vector<int> > &row_partitioning,
    const vector<int> &global_row_indices,
    const vector<int> &global_col_indices,
    const int &NUM_COLS_EFFECTIVE,
    map<int, CM_Hypers> &hypers_m,
    const vector<double> &ROW_CRP_ALPHA_GRID,
    const vector<double> &MULTINOMIAL_ALPHA_GRID,
    const vector<double> &R_GRID,
    const vector<double> &NU_GRID,
    const vector<double> &VM_B_GRID,
    const map<int, vector<double> > &S_GRIDS,
    const map<int, vector<double> > &MU_GRIDS,
    const map<int, vector<double> > &VM_A_GRIDS,
    const map<int, vector<double> > &VM_KAPPA_GRIDS,
    double CRP_ALPHA,
    int SEED) : crp_alpha(CRP_ALPHA), rng(SEED)
{
    crp_score = 0;
    data_score = 0;
    global_col_datatypes = GLOBAL_COL_DATATYPES;
    num_cols_effective = NUM_COLS_EFFECTIVE;
    //
    crp_alpha_grid = ROW_CRP_ALPHA_GRID;
    multinomial_alpha_grid = MULTINOMIAL_ALPHA_GRID;
    r_grid = R_GRID;
    nu_grid = NU_GRID;
    s_grids = S_GRIDS;
    mu_grids = MU_GRIDS;
    vm_b_grid = VM_B_GRID;
    vm_a_grids = VM_A_GRIDS;
    vm_kappa_grids = VM_KAPPA_GRIDS;
    //
    set_row_partitioning(row_partitioning);
    insert_cols(data, global_row_indices, global_col_indices, hypers_m);
}

// row partitioning unspecified, sample from crp
View::View(const MatrixD &data,
    const map<int, string> &GLOBAL_COL_DATATYPES,
    const vector<int> &global_row_indices,
    const vector<int> &global_col_indices,
    map<int, CM_Hypers> &hypers_m,
    const vector<double> &ROW_CRP_ALPHA_GRID,
    const vector<double> &MULTINOMIAL_ALPHA_GRID,
    const vector<double> &R_GRID,
    const vector<double> &NU_GRID,
    const vector<double> &VM_B_GRID,
    const map<int, vector<double> > &S_GRIDS,
    const map<int, vector<double> > &MU_GRIDS,
    const map<int, vector<double> > &VM_A_GRIDS,
    const map<int, vector<double> > &VM_KAPPA_GRIDS,
    int SEED) : rng(SEED)
{
    crp_score = 0;
    data_score = 0;
    global_col_datatypes = GLOBAL_COL_DATATYPES;
    //
    crp_alpha_grid = ROW_CRP_ALPHA_GRID;
    multinomial_alpha_grid = MULTINOMIAL_ALPHA_GRID;
    r_grid = R_GRID;
    nu_grid = NU_GRID;
    s_grids = S_GRIDS;
    mu_grids = MU_GRIDS;
    vm_b_grid = VM_B_GRID;
    vm_a_grids = VM_A_GRIDS;
    vm_kappa_grids = VM_KAPPA_GRIDS;
    //
    // sample alpha
    crp_alpha = crp_alpha_grid[draw_rand_i(crp_alpha_grid.size())];
    // sample partitioning
    set_row_partitioning(global_row_indices);
    // insert data
    insert_cols(data, global_row_indices, global_col_indices, hypers_m);
}

// empty View: for gibbs sampling a new partitioning of columns
View::View(const map<int, string> &GLOBAL_COL_DATATYPES,
    const vector<int> &global_row_indices,
    const vector<double> &ROW_CRP_ALPHA_GRID,
    const vector<double> &MULTINOMIAL_ALPHA_GRID,
    const vector<double> &R_GRID,
    const vector<double> &NU_GRID,
    const vector<double> &VM_B_GRID,
    const map<int, vector<double> > &S_GRIDS,
    const map<int, vector<double> > &MU_GRIDS,
    const map<int, vector<double> > &VM_A_GRIDS,
    const map<int, vector<double> > &VM_KAPPA_GRIDS,
    int SEED) : rng(SEED)
{
    crp_score = 0;
    data_score = 0;
    global_col_datatypes = GLOBAL_COL_DATATYPES;
    num_cols_effective = 0;
    //
    crp_alpha_grid = ROW_CRP_ALPHA_GRID;
    multinomial_alpha_grid = MULTINOMIAL_ALPHA_GRID;
    r_grid = R_GRID;
    nu_grid = NU_GRID;
    s_grids = S_GRIDS;
    mu_grids = MU_GRIDS;
    vm_b_grid = VM_B_GRID;
    vm_a_grids = VM_A_GRIDS;
    vm_kappa_grids = VM_KAPPA_GRIDS;
    //
    // sample alpha
    crp_alpha = crp_alpha_grid[draw_rand_i(crp_alpha_grid.size())];
    // sample partitioning
    set_row_partitioning(global_row_indices);
}

double View::get_num_vectors() const
{
    return cluster_lookup.size();
}

double View::get_num_cols() const
{
    return global_to_local.size();
}

int View::get_num_cols_effective() const
{
    return num_cols_effective;
}

int View::get_num_clusters() const
{
    return clusters.size();
}

double View::get_crp_score() const
{
    // return calc_crp_marginal();
    return crp_score;
}

double View::get_data_score() const
{
    // vector<Cluster*>::const_iterator it = clusters.begin();
    // double data_score = 0;
    // for(; it!=clusters.end(); ++it) {
    //   double data_score_i = (**it).get_marginal_logp();
    //   data_score += data_score_i;
    // }
    return data_score;
}

double View::get_score() const
{
    return get_crp_score() + get_data_score();
}

double View::get_crp_alpha() const
{
    return crp_alpha;
}

vector<double> View::get_crp_alpha_grid() const
{
    return crp_alpha_grid;
}

vector<string> View::get_hyper_strings(int which_col)
{
    vector<string> hyper_strings;
    int global_col_idx = get_key_of_value(global_to_local, which_col);
    // int global_col_idx = -1;
    // map<int, int>::const_iterator it;
    // for(it=global_to_local.begin(); it!=global_to_local.end(); ++it) {
    //   if(it->second==which_col) {
    //     global_col_idx = it->first;
    //   }
    // }
    string global_col_datatype = global_col_datatypes[global_col_idx];
    if (global_col_datatype == CONTINUOUS_DATATYPE) {
        hyper_strings.push_back("r");
        hyper_strings.push_back("nu");
        hyper_strings.push_back("s");
        hyper_strings.push_back("mu");
    } else if (global_col_datatype == CYCLIC_DATATYPE) {
        hyper_strings.push_back("a");
        hyper_strings.push_back("b");
        hyper_strings.push_back("kappa");
    } else if (global_col_datatype == MULTINOMIAL_DATATYPE) {
        hyper_strings.push_back("dirichlet_alpha");
    } else {
        cout << "View::get_hyper_strings(" << which_col <<
            "): invalid global_col_datatype: " << global_col_datatype << endl;
        assert(0);
    }
    return hyper_strings;
}

vector<double> View::get_hyper_grid(int global_col_idx,
    const string &which_hyper)
{
    vector<double> hyper_grid;
    if (which_hyper == "r") {
        hyper_grid = r_grid;
    } else if (which_hyper == "nu") {
        hyper_grid = nu_grid;
    } else if (which_hyper == "s") {
        hyper_grid = s_grids[global_col_idx];
    } else if (which_hyper == "mu") {
        hyper_grid = mu_grids[global_col_idx];
    } else if (which_hyper == "dirichlet_alpha") {
        hyper_grid = multinomial_alpha_grid;
    } else if (which_hyper == "a") {
        hyper_grid = vm_a_grids[global_col_idx];
    } else if (which_hyper == "b") {
        hyper_grid = vm_b_grid;
    } else if (which_hyper == "kappa") {
        hyper_grid = vm_kappa_grids[global_col_idx];
    } else {
        cout << "View::get_hyper_grid(" << global_col_idx << ", " << which_hyper <<
            "): invalid which_hyper" << endl;
        assert(0);
    }
    return hyper_grid;
}

CM_Hypers View::get_hypers(int local_col_idx) const
{
    return *(hypers_v[local_col_idx]);
}

map<string, double> View::get_row_partition_model_hypers() const
{
    CM_Hypers hypers;
    hypers["alpha"] = get_crp_alpha();
    return hypers;
}

vector<int> View::get_row_partition_model_counts() const
{
    return get_cluster_counts();
}

vector<map<string, double> > View::get_column_component_suffstats_i(
    int global_col_idx) const
{
    vector<map<string, double> > column_component_suffstats;
    vector<Cluster *>::const_iterator it = clusters.begin();
    for (; it != clusters.end(); ++it) {
        int local_col_idx = get(global_to_local, global_col_idx);
        map<string, double> suffstats = (**it).get_suffstats_i(local_col_idx);
        column_component_suffstats.push_back(suffstats);
    }
    return column_component_suffstats;
}

vector<vector<map<string, double> > > View::get_column_component_suffstats()
const
{
    vector<vector<map<string, double> > > column_component_suffstats;
    map<int, int>::const_iterator it;
    for (it = global_to_local.begin(); it != global_to_local.end(); ++it) {
        int global_col_idx = it->first;
        vector<map<string, double> > column_component_suffstats_i = \
            get_column_component_suffstats_i(global_col_idx);
        column_component_suffstats.push_back(column_component_suffstats_i);
    }
    return column_component_suffstats;
}

vector<int> View::get_global_col_indices()
{
    int num_cols = get_num_cols();
    vector<int> global_col_indices;
    for (int local_col_idx = 0; local_col_idx < num_cols; local_col_idx++) {
        int global_col_idx = get_key_of_value(global_to_local, local_col_idx);
        global_col_indices.push_back(global_col_idx);
    }
    return global_col_indices;
}

Cluster &View::get_cluster(int cluster_idx)
{
    assert(0 <= cluster_idx);
    assert((size_t)cluster_idx <= clusters.size());
    bool not_new = (size_t)cluster_idx < clusters.size();
    if (not_new) {
        return *clusters[cluster_idx];
    } else {
        return get_new_cluster();
    }
}

vector<double> View::get_draw(int row_idx, int random_seed) const
{
    Cluster &cluster = *get(cluster_lookup, row_idx);
    vector<double> draw = cluster.get_draw(random_seed);
    return draw;
}

vector<int> View::get_cluster_counts() const
{
    vector<int> counts;
    vector<Cluster *>::const_iterator it = clusters.begin();
    for (; it != clusters.end(); ++it) {
        int count = (**it).get_count();
        counts.push_back(count);
    }
    return counts;
}

double View::calc_cluster_vector_predictive_logp(const vector<double> &vd,
    const Cluster &which_cluster,
    double &crp_logp_delta,
    double &data_logp_delta) const
{
    int cluster_count = which_cluster.get_count();
    double score_delta;
    int num_vectors = get_num_vectors();
    // NOTE: non-mutating, so presume vector is not in state
    // so use num_vectors, not num_vectors - 1
    // should try to cache counts in some way
    crp_logp_delta = numerics::calc_cluster_crp_logp(cluster_count,
            num_vectors,
            crp_alpha);
    data_logp_delta = which_cluster.calc_row_predictive_logp(vd);
    score_delta = crp_logp_delta + data_logp_delta;
    return score_delta;
}

vector<double> View::calc_cluster_vector_predictive_logps(
    const vector<double> &vd)
{
    vector<double> logps;
    vector<Cluster *>::const_iterator it = clusters.begin();
    double crp_logp_delta, data_logp_delta;
    for (; it != clusters.end(); ++it) {
        logps.push_back(calc_cluster_vector_predictive_logp(vd, **it, crp_logp_delta,
                data_logp_delta));
    }
    Cluster empty_cluster(hypers_v);
    logps.push_back(calc_cluster_vector_predictive_logp(vd, empty_cluster,
            crp_logp_delta,
            data_logp_delta));
    empty_cluster.delete_component_models();
    return logps;
}

double View::calc_crp_marginal() const
{
    int num_vectors = get_num_vectors();
    vector<int> cluster_counts = get_cluster_counts();
    return numerics::calc_crp_alpha_conditional(cluster_counts, crp_alpha,
            num_vectors, true);
}

vector<double> View::calc_crp_marginals(const vector<double> &alphas_to_score)
const
{
    int num_vectors = get_num_vectors();
    vector<int> cluster_counts = get_cluster_counts();
    vector<double> crp_scores;
    vector<double>::const_iterator it = alphas_to_score.begin();
    for (; it != alphas_to_score.end(); ++it) {
        double alpha_to_score = *it;
        double this_crp_score = numerics::calc_crp_alpha_conditional(cluster_counts,
                alpha_to_score,
                num_vectors,
                true);
        crp_scores.push_back(this_crp_score);
    }
    return crp_scores;
}

vector<double> View::calc_hyper_conditionals(int which_col,
    const string &which_hyper,
    const vector<double> &hyper_grid) const
{
    vector<Cluster *>::const_iterator it;
    vector<vector<double> > vec_vec;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        vector<double> logps = (**it).calc_hyper_conditionals(which_col, which_hyper,
                hyper_grid);
        vec_vec.push_back(logps);
    }
    return std_vector_add(vec_vec);
}

double View::set_hyper(int which_col, const string &which_hyper,
    double new_value)
{
    vector<Cluster *>::const_iterator it;
    double score_delta = 0;
    // FIXME: this should use a mutator Cluster::set_hyper
    //        which in turn uses ComponentModel::set_hyper
    (*hypers_v[which_col])[which_hyper] = new_value;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        score_delta += (**it).incorporate_hyper_update(which_col);
    }
    // DOES THIS CAUSE UNBOUNDED SCORE GROWTH?
    data_score += score_delta;
    return score_delta;
}

double View::transition_hyper_i(int which_col, const string &which_hyper,
    const vector<double> &hyper_grid)
{
    //
    // draw new hyper
    vector<double> unorm_logps = calc_hyper_conditionals(which_col, which_hyper,
            hyper_grid);
    double rand_u = draw_rand_u();
    int draw = numerics::draw_sample_unnormalized(unorm_logps, rand_u);
    double new_hyper_value = hyper_grid[draw];
    //
    // update all clusters
    double score_delta = set_hyper(which_col, which_hyper, new_hyper_value);
    //
    return score_delta;
}

double View::transition_hyper_i(int which_col, const string &which_hyper)
{
    vector<int> global_ordering = extract_global_ordering(global_to_local);
    int global_col_idx = global_ordering[which_col];
    vector<double> hyper_grid = get_hyper_grid(global_col_idx, which_hyper);
    assert(hyper_grid.size() != 0);
    double score_delta = transition_hyper_i(which_col, which_hyper, hyper_grid);
    return score_delta;
}

double View::transition_hypers_i(int which_col)
{
    vector<string> hyper_strings = get_hyper_strings(which_col);
    random_shuffle(hyper_strings.begin(), hyper_strings.end(), rng);
    double score_delta = 0;
    vector<string>::iterator it;
    for (it = hyper_strings.begin(); it != hyper_strings.end(); ++it) {
        string which_hyper = *it;
        score_delta += transition_hyper_i(which_col, which_hyper);
    }
    return score_delta;
}

double View::transition_hypers()
{
    int num_cols = get_num_cols();
    double score_delta = 0;
    for (int local_col_idx = 0; local_col_idx < num_cols; local_col_idx++) {
        score_delta += transition_hypers_i(local_col_idx);
    }
    return score_delta;
}

double View::transition(const map<int, vector<double> > &row_data_map)
{
    vector<int> which_transitions = create_sequence(3);
    random_shuffle(which_transitions.begin(), which_transitions.end(), rng);
    double score_delta = 0;
    vector<int>::iterator it;
    for (it = which_transitions.begin(); it != which_transitions.end(); ++it) {
        int which_transition = *it;
        if (which_transition == 0) {
            score_delta += transition_hypers();
        } else if (which_transition == 1) {
            score_delta += transition_zs(row_data_map);
        } else if (which_transition == 2) {
            score_delta += transition_crp_alpha();
        } else {
            assert(0);
        }
    }
    return score_delta;
}

void View::increment_num_cols_effective()
{
    num_cols_effective++;
}

void View::decrement_num_cols_effective()
{
    num_cols_effective--;
}

double View::calc_column_predictive_logp(const vector<double> &column_data,
    const string &col_datatype,
    const vector<int> &data_global_row_indices,
    const CM_Hypers &hypers) const
{
    double score_delta = 0;
    vector<Cluster *>::const_iterator it;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        score_delta += (**it).calc_column_predictive_logp(column_data, col_datatype,
                data_global_row_indices,
                hypers);
    }
    return score_delta;
}

double View::set_crp_alpha(double new_crp_alpha)
{
    double crp_score_0 = crp_score;
    crp_alpha = new_crp_alpha;
    crp_score = calc_crp_marginal();
    return crp_score - crp_score_0;
}

Cluster &View::get_new_cluster()
{
    Cluster *p_new_cluster = new Cluster(hypers_v);
    clusters.push_back(p_new_cluster);
    return *p_new_cluster;
}

double View::insert_row(const vector<double> &vd, Cluster &which_cluster,
    int row_idx)
{
    // NOTE: MUST use calc_cluster_vector_logp,  gets crp_score_delta as well
    double crp_logp_delta, data_logp_delta;
    double score_delta = calc_cluster_vector_predictive_logp(vd, which_cluster,
            crp_logp_delta,
            data_logp_delta);
    which_cluster.insert_row(vd, row_idx);
    cluster_lookup[row_idx] = &which_cluster;
    crp_score += crp_logp_delta;
    data_score += data_logp_delta;
    return score_delta;
}

double View::insert_row(const vector<double> &vd,
    int matching_row_idx,
    int row_idx)
{
    Cluster &which_cluster = *cluster_lookup[matching_row_idx];
    double score_delta = insert_row(vd, which_cluster, row_idx);
    return score_delta;
}

double View::insert_row(const vector<double> &vd, int row_idx)
{
    vector<double> unorm_logps = calc_cluster_vector_predictive_logps(vd);
    double rand_u = draw_rand_u();
    int draw = numerics::draw_sample_unnormalized(unorm_logps, rand_u);
    Cluster &which_cluster = get_cluster(draw);
    double score_delta = insert_row(vd, which_cluster, row_idx);
    return score_delta;
}

double View::remove_row(const vector<double> &vd, int row_idx)
{
    Cluster &which_cluster = *(cluster_lookup[row_idx]);
    cluster_lookup.erase(cluster_lookup.find(row_idx));
    which_cluster.remove_row(vd, row_idx);
    double crp_logp_delta, data_logp_delta;
    double score_delta = calc_cluster_vector_predictive_logp(vd, which_cluster,
            crp_logp_delta,
            data_logp_delta);
    remove_if_empty(which_cluster);
    crp_score -= crp_logp_delta;
    data_score -= data_logp_delta;
    return score_delta;
}

void View::set_row_partitioning(const vector<vector<int> > &row_partitioning)
{
    int num_clusters = row_partitioning.size();
    vector<double> blank_row;
    for (int cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
        vector<int> global_row_indices = row_partitioning[cluster_idx];
        vector<int>::iterator it;
        // create new cluster
        Cluster &new_cluster = get_new_cluster();
        for (it = global_row_indices.begin(); it != global_row_indices.end(); ++it) {
            int global_row_index = *it;
            insert_row(blank_row, new_cluster, global_row_index);
        }
    }
}

void View::set_row_partitioning(const vector<int> &global_row_indices)
{
    vector<vector<int> > crp_init = draw_crp_init(global_row_indices, crp_alpha,
            rng);
    set_row_partitioning(crp_init);
}

double View::insert_col(const vector<double> &col_data,
    const vector<int> &data_global_row_indices,
    int global_col_idx,
    CM_Hypers &hypers)
{
    double score_delta = 0;
    string col_datatype = global_col_datatypes[global_col_idx];
    //
    hypers_v.push_back(&hypers);
    vector<Cluster *>::const_iterator it;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        score_delta += (**it).insert_col(col_data, col_datatype,
                data_global_row_indices, hypers);
    }
    int num_cols = get_num_cols();
    global_to_local[global_col_idx] = num_cols;
    data_score += score_delta;
    return score_delta;
}

double View::insert_cols(const MatrixD &data,
    const vector<int> &global_row_indices,
    const vector<int> &global_col_indices,
    map<int, CM_Hypers> &hypers_m)
{
    int num_cols = global_col_indices.size();
    double score_delta = 0;
    for (int data_col_idx = 0; data_col_idx < num_cols; data_col_idx++) {
        vector<double> col_data = extract_col(data, data_col_idx);
        int global_col_idx = global_col_indices[data_col_idx];
        CM_Hypers &hypers = hypers_m[global_col_idx];
        score_delta += insert_col(col_data, global_row_indices, global_col_idx,
                hypers);
    }
    return score_delta;
}

double View::remove_col(int global_col_idx)
{
    // FIXME: should pop hyper_grid elements
    int local_col_idx = global_to_local[global_col_idx];
    //
    vector<Cluster *>::const_iterator it;
    double score_delta = 0;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        score_delta += (*it)->remove_col(local_col_idx);
    }
    // rearrange global_to_local
    vector<int> global_col_indices = extract_global_ordering(global_to_local);
    global_col_indices.erase(global_col_indices.begin() + local_col_idx);
    hypers_v.erase(hypers_v.begin() + local_col_idx);
    global_to_local = construct_lookup_map(global_col_indices);
    //
    data_score -= score_delta;
    return score_delta;
}

void View::remove_if_empty(Cluster &which_cluster)
{
    if (which_cluster.get_count() == 0) {
        vector<Cluster *>::iterator it;
        for (it = clusters.begin(); it != clusters.end(); ++it) {
            if (*it == &which_cluster) {
                clusters.erase(it);
                which_cluster.delete_component_models();
                delete &which_cluster;
                break;
            }
        }
    }
}

void View::remove_all()
{
    cluster_lookup.clear();
    vector<Cluster *>::const_iterator it = clusters.begin();
    for (; it != clusters.end(); ++it) {
        Cluster &which_cluster = **it;
        which_cluster.delete_component_models(false);
        delete &which_cluster;
    }
    clusters.resize(0);
}

double View::transition_z(const vector<double> &vd, int row_idx)
{
    double score_delta = 0;
    score_delta += remove_row(vd, row_idx);
    score_delta += insert_row(vd, row_idx);
    return score_delta;
}

double View::transition_zs(const map<int, vector<double> > &row_data_map)
{
    double score_delta = 0;
    vector<int> shuffled_row_indices = shuffle_row_indices();
    vector<int>::iterator it = shuffled_row_indices.begin();
    for (; it != shuffled_row_indices.end(); ++it) {
        int row_idx = *it;
        const vector<double> &vd = get(row_data_map, row_idx);
        score_delta += transition_z(vd, row_idx);
    }
    return score_delta;
}

double View::transition_crp_alpha()
{
    // to make score_crp not calculate absolute, need to track score deltas
    // and apply delta to crp_score
    double crp_score_0 = get_crp_score();
    vector<double> unorm_logps = calc_crp_marginals(crp_alpha_grid);
    double rand_u = draw_rand_u();
    int draw = numerics::draw_sample_unnormalized(unorm_logps, rand_u);
    crp_alpha = crp_alpha_grid[draw];
    crp_score = unorm_logps[draw];
    double crp_score_delta = crp_score - crp_score_0;
    return crp_score_delta;
}

vector<double> View::align_data(const vector<double> &raw_values,
    const vector<int> &global_column_indices) const
{
    return reorder_per_map(raw_values, global_column_indices, global_to_local);
}

vector<int> View::shuffle_row_indices()
{
    // can't use std::random_shuffle b/c need to control seed
    vector<int> original_order;
    map<int, Cluster *>::iterator it = cluster_lookup.begin();
    for (; it != cluster_lookup.end(); ++it) {
        original_order.push_back(it->first);
    }
    vector<int> shuffled_order;
    while (original_order.size() != 0) {
        int draw = draw_rand_i(original_order.size());
        int row_idx = original_order[draw];
        shuffled_order.push_back(row_idx);
        original_order.erase(original_order.begin() + draw);
    }
    return shuffled_order;
}

vector<vector<int> > View::get_cluster_groupings() const
{
    vector<vector<int> > cluster_groupings;
    vector<Cluster *>::const_iterator it;
    for (it = clusters.begin(); it != clusters.end(); ++it) {
        Cluster &c = **it;
        vector<int> row_indices = c.get_row_indices_vector();
        cluster_groupings.push_back(row_indices);
    }
    return cluster_groupings;
}

vector<int> View::get_canonical_clustering() const
{
    map<Cluster *, int> view_to_int = vector_to_map(clusters);
    vector<int> canonical_clustering;
    for (unsigned int i = 0; i < cluster_lookup.size(); i++) {
        Cluster *p_c = cluster_lookup.find(i)->second;
        int canonical_cluster_idx = view_to_int[p_c];
        canonical_clustering.push_back(canonical_cluster_idx);
    }
    return canonical_clustering;
}

ostream &operator<<(ostream &os, const View &v)
{
    os << v.to_string() << endl;
    return os;
}

string View::to_string(const string &join_str, bool top_level) const
{
    stringstream ss;
    if (!top_level) {
        vector<Cluster *>::const_iterator it = clusters.begin();
        int cluster_idx = 0;
        for (; it != clusters.end(); ++it) {
            ss << "CLUSTER IDX: " << cluster_idx++ << join_str;
            ss << **it << endl;
        }
    }
    ss << "global_to_local: " << global_to_local << join_str;
    ss << "crp_score: " << get_crp_score();
    ss << ", " << "data_score: " << get_data_score();
    ss << ", " << "score: " << get_score();
    return ss.str();
}

void View::print_score_matrix()
{
    vector<vector<double> > scores_v;
    vector<Cluster *>::const_iterator c_it;
    for (c_it = clusters.begin(); c_it != clusters.end(); ++c_it) {
        Cluster &c = **c_it;
        vector<double> scores = c.calc_marginal_logps();
        scores_v.push_back(scores);
    }
    cout << "crp_score: " << crp_score << endl;
    cout << "scores_v:" << endl << scores_v << endl;
}

void View::print()
{
    cout << to_string() << endl;
}

//void View::assert_state_consistency() {
//  double tolerance = 1E-10;
//  vector<int> cluster_counts = get_cluster_counts();
//  int sum_via_cluster_counts = std::accumulate(cluster_counts.begin(),
//                         cluster_counts.end(), 0);
//  assert(is_almost(get_num_vectors(), sum_via_cluster_counts, tolerance));
//  assert(is_almost(get_crp_score(),calc_crp_marginal(), tolerance));
//}

double View::draw_rand_u()
{
    return rng.next();
}

int View::draw_rand_i(int max)
{
    return rng.nexti(max);
}
