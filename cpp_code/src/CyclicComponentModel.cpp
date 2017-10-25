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
#include "CyclicComponentModel.h"
using namespace std;

CyclicComponentModel::CyclicComponentModel(const CM_Hypers &in_hypers)
{
    count = 0;
    score = 0;
    p_hypers = &in_hypers;
    hyper_kappa = get(*p_hypers, string("kappa"));
    hyper_a = get(*p_hypers, string("a"));
    hyper_b = get(*p_hypers, string("b"));
    init_suffstats();
    set_log_Z_0();
}

CyclicComponentModel::CyclicComponentModel(const CM_Hypers &in_hypers,
    int COUNT, double SUM_SIN_X, double SUM_COS_X)
{
    count = COUNT;
    sum_sin_x = SUM_SIN_X;
    sum_cos_x = SUM_COS_X;
    p_hypers = &in_hypers;
    hyper_kappa = get(*p_hypers, string("kappa"));
    hyper_a = get(*p_hypers, string("a"));
    hyper_b = get(*p_hypers, string("b"));
    set_log_Z_0();
    score = calc_marginal_logp();
}


void CyclicComponentModel::get_hyper_doubles(double &kappa, double &a,
    double &b) const
{
    kappa = hyper_kappa;
    a = hyper_a;
    b = hyper_b;
}

double CyclicComponentModel::calc_marginal_logp() const
{
    double kappa, a, b;
    int count;
    double sum_sin_x, sum_cos_x;
    get_hyper_doubles(kappa, a, b);
    double Z0 = numerics::calc_cyclic_log_Z(a);
    get_suffstats(count, sum_sin_x, sum_cos_x);
    numerics::update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa, a, b);
    // return numerics::calc_cyclic_logp(count, kappa, a, log_Z_0);
    return numerics::calc_cyclic_logp(count, kappa, a, Z0);
}

double CyclicComponentModel::calc_element_predictive_logp(
    double element) const
{
    if (isnan(element)) {
        return 0;
    }
    double kappa, a, b;
    int count;
    double sum_sin_x, sum_cos_x;
    get_hyper_doubles(kappa, a, b);
    get_suffstats(count, sum_sin_x, sum_cos_x);
    //
    // numerics::insert_to_cyclic_suffstats(count, sum_sin_x, sum_cos_x, element);
    // numerics::update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa, a, b);
    double logp_prime = numerics::calc_cyclic_data_logp(count, sum_sin_x, sum_cos_x,
            kappa, a, b, element);
    return logp_prime;
}

double CyclicComponentModel::calc_element_predictive_logp_constrained(
    double element, const vector<double> &constraints) const
{
    if (isnan(element)) {
        return 0;
    }
    double kappa, a, b;
    int count;
    double sum_sin_x, sum_cos_x;
    get_hyper_doubles(kappa, a, b);
    get_suffstats(count, sum_sin_x, sum_cos_x);
    //
    int num_constraints = (int) constraints.size();
    for (int constraint_idx = 0; constraint_idx < num_constraints;
        constraint_idx++) {
        double constraint = constraints[constraint_idx];
        numerics::insert_to_cyclic_suffstats(count, sum_sin_x, sum_cos_x, constraint);
    }
    // numerics::update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa, a, b);
    // double baseline = numerics::calc_cyclic_logp(count, kappa, a, log_Z_0);
    //
    get_hyper_doubles(kappa, a, b);
    // numerics::insert_to_cyclic_suffstats(count, sum_sin_x, sum_cos_x, element);
    // numerics::update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa, a, b);
    // double updated = numerics::calc_cyclic_data_logp(count, kappa, a, log_Z_0);
    double predictive_logp = numerics::calc_cyclic_data_logp(count, sum_sin_x,
            sum_cos_x,
            kappa, a, b, element);
    return predictive_logp;
}

vector<double> CyclicComponentModel::calc_hyper_conditionals(
    const string &which_hyper, const vector<double> &hyper_grid) const
{
    double kappa, a, b;
    int count;
    double sum_sin_x, sum_cos_x;
    get_hyper_doubles(kappa, a, b);
    get_suffstats(count, sum_sin_x, sum_cos_x);
    if (which_hyper == "a") {
        return numerics::calc_cyclic_a_conditionals(hyper_grid, count, sum_sin_x,
                sum_cos_x, kappa, b);
    } else if (which_hyper == "b") {
        return numerics::calc_cyclic_b_conditionals(hyper_grid, count, sum_sin_x,
                sum_cos_x, kappa, a);
    } else if (which_hyper == "kappa") {
        return numerics::calc_cyclic_kappa_conditionals(hyper_grid, count, sum_sin_x,
                sum_cos_x, a, b);
    } else {
        // error condition
        vector<double> error;
        return error;
    }
}

double CyclicComponentModel::insert_element(double element)
{
    if (isnan(element)) {
        return 0;
    }
    double score_0 = score;
    numerics::insert_to_cyclic_suffstats(count, sum_sin_x, sum_cos_x, element);
    score = calc_marginal_logp();
    double delta_score = score - score_0;
    return delta_score;
}

double CyclicComponentModel::remove_element(double element)
{
    if (isnan(element)) {
        return 0;
    }
    double score_0 = score;
    numerics::remove_from_cyclic_suffstats(count, sum_sin_x, sum_cos_x, element);
    score = calc_marginal_logp();
    double delta_score = score - score_0;
    return delta_score;
}

double CyclicComponentModel::incorporate_hyper_update()
{
    hyper_kappa = get(*p_hypers, (string) "kappa");
    hyper_a = get(*p_hypers, (string) "a");
    hyper_b = get(*p_hypers, (string) "b");
    double score_0 = score;
    // hypers[which_hyper] = value; // set by owner of hypers object
    set_log_Z_0();
    score = calc_marginal_logp();
    double score_delta = score - score_0;
    return score_delta;
}

void CyclicComponentModel::set_log_Z_0()
{
    double kappa, a, b;
    get_hyper_doubles(kappa, a, b);
    log_Z_0 = numerics::calc_cyclic_log_Z(a);
}

void CyclicComponentModel::init_suffstats()
{
    sum_sin_x = 0.;
    sum_cos_x = 0.;
}

void CyclicComponentModel::get_suffstats(int &count_out, double &sum_sin_x_out,
    double &sum_cos_x_out) const
{
    count_out = count;
    sum_sin_x_out = sum_sin_x;
    sum_cos_x_out = sum_cos_x;
}

double CyclicComponentModel::get_draw(int random_seed) const
{
    vector<double> constraints;
    return get_draw_constrained(random_seed, constraints);
}

double CyclicComponentModel::get_draw_constrained(int random_seed,
    const vector<double> &constraints) const
{
    // get modified suffstats
    double kappa, a, b;
    int count;
    double sum_sin_x, sum_cos_x;
    get_hyper_doubles(kappa, a, b);
    get_suffstats(count, sum_sin_x, sum_cos_x);
    int num_constraints = (int) constraints.size();
    for (int constraint_idx = 0; constraint_idx < num_constraints;
        constraint_idx++) {
        double constraint = constraints[constraint_idx];
        numerics::insert_to_cyclic_suffstats(count, sum_sin_x, sum_cos_x, constraint);
    }
    numerics::update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa, a, b);
    // Rejection sampling.
    // we need to get a good estimate of a constant M to containthe etire pdf
    // but not to reject too many samples. We derive it from the posterior update
    // parameters
    // Proposal distribution is uniform scaled to the the max value of the
    // predictive pdf
    // FIXME: This will lead to a lot of rejections especially for high kappa
    RandomNumberGenerator gen(random_seed);
    bool rejected = true;
    double x; // random number
    double l_p;   // log proposal value
    double pdf_t; // log predictive probability
    double log_M = calc_element_predictive_logp_constrained(b, constraints);
    unsigned short int itr = 0;
    while (rejected && itr < 1000) {
        // generate random number in domain from proposal distribution
        x = gen.next() * 2 * M_PI;
        l_p = log(gen.next()) + log_M;
        // get pdf at target
        pdf_t = calc_element_predictive_logp_constrained(x, constraints);
        if (l_p < pdf_t) {
            rejected = false;
            return x;
        }
        ++itr;
    }
    assert(false);
    return 0;         // XXXGCC
}

map<string, double> CyclicComponentModel::_get_suffstats() const
{
    map<string, double> suffstats;
    suffstats["sum_sin_x"] = sum_sin_x;
    suffstats["sum_cos_x"] = sum_cos_x;
    return suffstats;
}

map<string, double> CyclicComponentModel::get_hypers() const
{
    map<string, double> hypers;
    hypers["kappa"] = hyper_kappa;
    hypers["a"] = hyper_a;
    hypers["b"] = hyper_b;
    return hypers;
}

