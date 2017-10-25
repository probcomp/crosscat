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
#ifndef GUARD_numerics_h
#define GUARD_numerics_h

#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>
#include "utils.h"

static const double LOG_2PI = log(2.0 * M_PI);
static const double HALF_LOG_2PI = .5 * LOG_2PI;
static const double LOG_2 = log(2.0);

// use a namespce to hold all the functions?
// http://stackoverflow.com/questions/6108704/renaming-namespaces

namespace numerics
{

// signum
template <typename T> int sgn(T val);

double estimate_vonmises_kappa(const std::vector<double> &X);

double i_0(double x);
double i_1(double x);

double log_bessel_0(double x); // log I_0(x)

double logaddexp(const std::vector<double> &logs);

// sampling given vector of logps or related
int draw_sample_unnormalized(const std::vector<double> &unorm_logps,
    double rand_u);
int draw_sample_with_partition(const std::vector<double> &unorm_logps,
    double log_partition, double rand_u);
int crp_draw_sample(const std::vector<int> &counts, int sum_counts,
    double alpha, double rand_u);

// crp probability functions
double calc_cluster_crp_logp(double cluster_weight, double sum_weights,
    double alpha);
double calc_crp_alpha_conditional(const std::vector<int> &counts, double alpha,
    int sum_counts = -1, bool absolute = false);
std::vector<double> calc_crp_alpha_conditionals(const std::vector<double> &grid,
    const std::vector<int> &counts,
    bool absolute = false);

// continuous suffstats functions
//
//   mutators
void insert_to_continuous_suffstats(int &count,
    double &sum_x, double &sum_x_sq,
    double el);
void remove_from_continuous_suffstats(int &count,
    double &sum_x, double &sum_x_sq,
    double el);
void update_continuous_hypers(int count,
    double sum_x, double sum_x_sq,
    double &r, double &nu,
    double &s, double &mu);
//   calculators
double calc_continuous_logp(int count,
    double r, double nu, double s,
    double log_Z_0);
double calc_continuous_data_logp(int count,
    double sum_x, double sum_x_sq,
    double r, double nu,
    double s, double mu,
    double el,
    double score_0);
std::vector<double> calc_continuous_r_conditionals(
    const std::vector<double> &r_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double nu,
    double s,
    double mu);
std::vector<double> calc_continuous_nu_conditionals(
    const std::vector<double> &nu_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double r,
    double s,
    double mu);
std::vector<double> calc_continuous_s_conditionals(
    const std::vector<double> &s_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double r,
    double nu,
    double mu);
std::vector<double> calc_continuous_mu_conditionals(
    const std::vector<double> &mu_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double r,
    double nu,
    double s);

// multinomial suffstats functions
//
//   mutators (NONE FOR NOW)
//
// calculators
double calc_multinomial_marginal_logp(int count,
    const std::vector<int> &counts,
    int K,
    double dirichlet_alpha);
double calc_multinomial_predictive_logp(double element,
    const std::vector<int> &counts,
    int sum_counts,
    int K, double dirichlet_alpha);
std::vector<double> calc_multinomial_dirichlet_alpha_conditional(
    const std::vector<double> &dirichlet_alpha_grid,
    int count,
    const std::vector<int> &counts,
    int K);

// cyclic component model functions
//
// mutators
void insert_to_cyclic_suffstats(int &count,
    double &sum_cos_x, double &sum_sin_x,
    double el);
void remove_from_cyclic_suffstats(int &count,
    double &sum_sin_x, double &sum_cos_x,
    double el);
void update_cyclic_hypers(int count,
    double sum_sin_x, double sum_cos_x,
    double kappa, double &a, double &b);

// calculators
double calc_cyclic_log_Z(double a);
double calc_cyclic_logp(int count, double kappa, double a, double log_Z_0);
double calc_cyclic_data_logp(int count,
    double sum_sin_x, double sum_cos_x,
    double kappa, double a, double b,
    double el);

std::vector<double> calc_cyclic_a_conditionals(
    const std::vector<double> &a_grid,
    int count,
    double sum_sin_x,
    double sum_cos_x,
    double kappa,
    double b);
std::vector<double> calc_cyclic_b_conditionals(
    const std::vector<double> &b_grid,
    int count,
    double sum_sin_x,
    double sum_cos_x,
    double kappa,
    double a);
std::vector<double> calc_cyclic_kappa_conditionals(const std::vector<double>
    &kappa_grid,
    int count,
    double sum_sin_x,
    double sum_cos_x,
    double a,
    double b);

} // namespace numerics

#endif //GUARD_numerics_h
