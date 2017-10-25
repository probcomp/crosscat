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

#include "MultinomialComponentModel.h"

using namespace std;

MultinomialComponentModel::MultinomialComponentModel(const CM_Hypers
    &in_hypers)
{
    count = 0;
    score = 0;
    p_hypers = &in_hypers;
    hyper_K = get(*p_hypers, (string) "K");
    hyper_dirichlet_alpha = get(*p_hypers, (string) "dirichlet_alpha");
    init_suffstats();
    set_log_Z_0();
}

MultinomialComponentModel::MultinomialComponentModel(const CM_Hypers &in_hypers,
    int count_in,
    const map<string, double> &counts)
{
    count = 0;
    score = 0;
    p_hypers = &in_hypers;
    hyper_K = get(*p_hypers, (string) "K");
    hyper_dirichlet_alpha = get(*p_hypers, (string) "dirichlet_alpha");
    set_log_Z_0();
    // set suffstats
    count = count_in;
    init_suffstats();
    for (map<string, double>::const_iterator it = counts.begin();
        it != counts.end();
        ++it) {
        int i = intify(it->first);
        assert(0 <= i);
        assert(i < hyper_K);
        assert(0 <= it->second);
        assert(it->second == trunc(it->second));
        suffstats[i] = static_cast<int>(it->second);
    }
    score = calc_marginal_logp();
}

void MultinomialComponentModel::init_suffstats()
{
    suffstats.resize(hyper_K);
}

double MultinomialComponentModel::calc_marginal_logp() const
{
    const vector<int> &counts = suffstats;
    int K = hyper_K;
    double dirichlet_alpha = hyper_dirichlet_alpha;
    return numerics::calc_multinomial_marginal_logp(count, counts, K,
            dirichlet_alpha);
}

double MultinomialComponentModel::calc_element_predictive_logp(
    double element) const
{
    if (isnan(element)) {
        return 0;
    }
    int K = hyper_K;
    double dirichlet_alpha = hyper_dirichlet_alpha;
    double logp = numerics::calc_multinomial_predictive_logp(element,
            suffstats, count,
            K, dirichlet_alpha);
    return logp;
}

double MultinomialComponentModel::calc_element_predictive_logp_constrained(
    double element, const vector<double> &constraints) const
{
    if (isnan(element)) {
        return 0;
    }
    int K = hyper_K;
    double dirichlet_alpha = hyper_dirichlet_alpha;
    //
    vector<int> suffstats_copy = suffstats;
    int count_copy = count;
    int num_constraints = (int) constraints.size();
    for (int constraint_idx = 0; constraint_idx < num_constraints;
        constraint_idx++) {
        double constraint = constraints[constraint_idx];
        assert(0 <= constraint);
        assert(constraint < K);
        assert(constraint == trunc(constraint));
        int i = static_cast<int>(constraint);
        count_copy++;
        suffstats_copy[i]++;
    }
    double predictive = \
        numerics::calc_multinomial_predictive_logp(element,
            suffstats_copy,
            count_copy,
            K, dirichlet_alpha);
    return predictive;
}

vector<double> MultinomialComponentModel::calc_hyper_conditionals(
    const string &which_hyper, const vector<double> &hyper_grid) const
{
    const vector<int> &counts = suffstats;
    int K = hyper_K;
    if (which_hyper == "dirichlet_alpha") {
        return numerics::calc_multinomial_dirichlet_alpha_conditional(hyper_grid,
                count,
                counts,
                K);
    } else {
        // error condition
        cout << "MultinomialComponentModel::calc_hyper_conditional: bad value for which_hyper="
            << which_hyper << endl;
        assert(0);
        vector<double> vd;
        return vd;
    }
}

double MultinomialComponentModel::insert_element(double element)
{
    if (isnan(element)) {
        return 0;
    }
    assert(element == trunc(element));
    int i = static_cast<int>(element);
    double delta_score = calc_element_predictive_logp(element);
    suffstats[i] += 1;
    count += 1;
    score += delta_score;
    return delta_score;
}

double MultinomialComponentModel::remove_element(double element)
{
    if (isnan(element)) {
        return 0;
    }
    assert(element == trunc(element));
    int i = static_cast<int>(element);
    assert(0 < suffstats[i]);
    suffstats[i] -= 1;
    double delta_score = calc_element_predictive_logp(element);
    count -= 1;
    score -= delta_score;
    return delta_score;
}

double MultinomialComponentModel::incorporate_hyper_update()
{
    hyper_K = get(*p_hypers, (string) "K");
    hyper_dirichlet_alpha = get(*p_hypers, (string) "dirichlet_alpha");
    double score_0 = score;
    // hypers[which_hyper] = value; // set by owner of hypers object
    score = calc_marginal_logp();
    double score_delta = score - score_0;
    return score_delta;
}

void MultinomialComponentModel::set_log_Z_0()
{
    log_Z_0 = calc_marginal_logp();
}

void MultinomialComponentModel::get_keys_counts_for_draw(vector<int> &keys,
    vector<double> &log_counts_for_draw,
    const vector<int> &counts) const
{
    double dirichlet_alpha = hyper_dirichlet_alpha;
    for (int key = 0; key < hyper_K; key++) {
        int count_for_draw = counts[key];
        // "update" counts by adding dirichlet alpha to each value
        count_for_draw += dirichlet_alpha;
        keys.push_back(key);
        log_counts_for_draw.push_back(log(count_for_draw));
    }
    return;
}

double MultinomialComponentModel::get_draw(int random_seed) const
{
    // get modified suffstats
    const vector<int> &counts = suffstats;
    // get a random draw
    double uniform_draw = RandomNumberGenerator(random_seed).next();
    //
    vector<int> keys;
    vector<double> log_counts_for_draw;
    get_keys_counts_for_draw(keys, log_counts_for_draw, counts);
    //
    int key_idx = numerics::draw_sample_unnormalized(log_counts_for_draw,
            uniform_draw);
    double draw = static_cast<double>(keys[key_idx]);
    return draw;
}

double MultinomialComponentModel::get_draw_constrained(int random_seed,
    const vector<double> &constraints) const
{
    // get modified suffstats
    const vector<int> &counts = suffstats;
    // get a random draw
    double uniform_draw = RandomNumberGenerator(random_seed).next();
    //
    vector<int> keys;
    vector<double> log_counts_for_draw;
    get_keys_counts_for_draw(keys, log_counts_for_draw, counts);
    map<int, int> index_lookup = construct_lookup_map(keys);
    int num_constraints = (int) constraints.size();
    for (int constraint_idx = 0; constraint_idx < num_constraints;
        constraint_idx++) {
        double constraint = constraints[constraint_idx];
        assert(0 <= constraint);
        assert(constraint == trunc(constraint));
        int count_idx = index_lookup[static_cast<int>(constraint)];
        double log_count = log_counts_for_draw[count_idx];
        log_counts_for_draw[count_idx] = log(exp(log_count) + 1);
    }
    //
    int key_idx = numerics::draw_sample_unnormalized(log_counts_for_draw,
            uniform_draw);
    double draw = static_cast<double>(keys[key_idx]);
    return draw;
}

map<string, double> MultinomialComponentModel::_get_suffstats() const
{
    map<string, double> counts;
    for (int key = 0; key < hyper_K; key++) {
        counts[stringify(key)] = suffstats[key];
    }
    return counts;
}

map<string, double> MultinomialComponentModel::get_hypers() const
{
    map<string, double> hypers;
    hypers["K"] = hyper_K;
    hypers["dirichlet_alpha"] = hyper_dirichlet_alpha;
    return hypers;
}

void MultinomialComponentModel::get_suffstats(int &count_out,
    map<string, double> &counts) const
{
    count_out = count;
    counts = _get_suffstats();
}
