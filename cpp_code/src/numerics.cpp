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
#include "numerics.h"

using namespace std;

namespace numerics
{

// return sign of val (signum function)
template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

#define arraycount(A) (sizeof(A)/sizeof(*A))

double polyeval(const double a[], size_t n, double x)
{
    size_t i = n;
    double y = a[--i];
    while (0 < i--) {
        y *= x;
        y += a[i];
    }
    return y;
}

// Modified Bessel function of the first kind, at 0 and 1, derived
// from
//
//      J.M. Blair and C.A. Edwards, `Stable rational minimax
//      approximations to the modified Bessel functions I_0(x) and
//      I_1(x)', AECL-4928, Chalk River Nuclear Laboratories, Chalk
//      River, Ontario, October 1974.
//
//      http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/06/178/6178667.pdf

double i_0(double x)
{
    static const double P1[] = {
        -2.2335582639474375249e+15,
        -5.5050369673018427753e+14,
        -3.2940087627407749166e+13,
        -8.4925101247114157499e+11,
        -1.1912746104985237192e+10,
        -1.0313066708737980747e+08,
        -5.9545626019847898221e+05,
        -2.4125195876041896775e+03,
        -7.0935347449210549190e+00,
        -1.5453977791786851041e-02,
        -2.5172644670688975051e-05,
        -3.0517226450451067446e-08,
        -2.6843448573468483278e-11,
        -1.5982226675653184646e-14,
        -5.2487866627945699800e-18,
    };
    static const double Q1[] = {
        -2.2335582639474375245e+15,
        +7.8858692566751002988e+12,
        -1.2207067397808979846e+10,
        +1.0377081058062166144e+07,
        -4.8527560179962773045e+03,
        +1.0,
    };
    static const double P2[] = {
        -2.2210262233306573296e-04,
        +1.3067392038106924055e-02,
        -4.4700805721174453923e-01,
        +5.5674518371240761397e+00,
        -2.3517945679239481621e+01,
        +3.1611322818701131207e+01,
        -9.6090021968656180000e+00,
    };
    static const double Q2[] = {
        -5.5194330231005480228e-04,
        +3.2547697594819615062e-02,
        -1.1151759188741312645e+00,
        +1.3982595353892851542e+01,
        -6.0228002066743340583e+01,
        +8.5539563258012929600e+01,
        -3.1446690275135491500e+01,
        +1.0,
    };
    double y, q, v;
    if (x < 0) {
        x = -x;
    }
    if (x == 0) {
        return 1;
    }
    if (x <= 15) {
        y = x * x;
        q = polyeval(P1, arraycount(P1), y) / polyeval(Q1, arraycount(Q1), y);
        v = q;
    } else {
        y = 1 / x - static_cast<double>(1) / 15;
        q = polyeval(P2, arraycount(P2), y) / polyeval(Q2, arraycount(Q2), y);
        v = (exp(x) / sqrt(x)) * q;
    }
    return v;
}

double i_1(double x)
{
    static const double P1[] = {
        -1.4577180278143463643e+15,
        -1.7732037840791591320e+14,
        -6.9876779648010090070e+12,
        -1.3357437682275493024e+11,
        -1.4828267606612366099e+09,
        -1.0588550724769347106e+07,
        -5.1894091982308017540e+04,
        -1.8225946631657315931e+02,
        -4.7207090827310162436e-01,
        -9.1746443287817501309e-04,
        -1.3466829827635152875e-06,
        -1.4831904935994647675e-09,
        -1.1928788903603238754e-12,
        -6.5245515583151902910e-16,
        -1.9705291802535139930e-19,
    };
    static const double Q1[] = {
        -2.9154360556286927285e+15,
        +9.7887501377547640438e+12,
        -1.4386907088588283434e+10,
        +1.1594225856856884006e+07,
        -5.1326864679904189920e+03,
        +1.0,
    };
    static const double P2[] = {
        +1.4582087408985668208e-05,
        -8.9359825138577646443e-04,
        +2.9204895411257790122e-02,
        -3.4198728018058047439e-01,
        +1.3960118277609544334e+00,
        -1.9746376087200685843e+00,
        +8.5591872901933459000e-01,
        -6.0437159056137599999e-02,
    };
    static const double Q2[] = {
        +3.7510433111922824643e-05,
        -2.2835624489492512649e-03,
        +7.4212010813186530069e-02,
        -8.5017476463217924408e-01,
        +3.2593714889036996297e+00,
        -3.8806586721556593450e+00,
        +1.0,
    };
    double xabs, y, q, v;
    if (x == 0) {
        return 0;
    }
    xabs = fabs(x);
    if (xabs <= 15) {
        y = x * x;
        q = polyeval(P1, arraycount(P1), y) / polyeval(Q1, arraycount(Q1), y);
        v = xabs * q;
    } else {
        y = 1 / xabs - static_cast<double>(1) / 15;
        q = polyeval(P2, arraycount(P2), y) / polyeval(Q2, arraycount(Q2), y);
        v = (exp(xabs) / sqrt(xabs)) * q;
    }
    if (x < 0) {
        v = -v;
    }
    return v;
}

double estimate_vonmises_kappa(const vector<double> &X)
{
    // Newton's method solution for ML estimate of kappa
    double N = (double) X.size();
    double sum_sin_x = 0;
    double sum_cos_x = 0;
    vector<double>::const_iterator it = X.begin();
    for (; it != X.end(); it++) {
        double x = *it;
        sum_sin_x += sin(x);
        sum_cos_x += cos(x);
    }
    double R2 = (sum_sin_x / N) * (sum_sin_x / N) + (sum_cos_x / N) *
        (sum_cos_x / N);
    double R = sqrt(R2) ;
    double kappa = R * (2.0 - R2) / (1.0 - R2);
    double Ap, kappa_1;
    Ap = i_1(kappa) / i_0(kappa);
    kappa_1 = kappa - (Ap - R) / (1 - Ap * Ap - Ap / kappa);
    Ap = i_1(kappa_1) / i_0(kappa_1);
    kappa = kappa_1 - (Ap - R) / (1 - Ap * Ap - Ap / kappa_1);
    Ap = i_1(kappa) / i_0(kappa);
    kappa_1 = kappa - (Ap - R) / (1 - Ap * Ap - Ap / kappa);
    Ap = i_1(kappa_1) / i_0(kappa_1);
    kappa = kappa_1 - (Ap - R) / (1 - Ap * Ap - Ap / kappa_1);
    return (kappa > 0) ? kappa : 1.0 / X.size();
}

double log_bessel_0(double x)
{
    double i0 = i_0(x);
    // As x grows without bound, i_0(x) ---> e^x / sqrt(2 pi x).  So if
    // naive calculation of i_0 overflows, approximate it with the
    // asymptotic expansion.  See NIST DLMF 10.40(i) (Modified Bessel
    // Functions, Asymptotic Expansions for Large Argument, Hankel's
    // Expansions), Eq. (10.40.1) <http://dlmf.nist.gov/10.40.E1>.  We
    // assume the summation to be negligible if x is large enough to
    // overflow.
    if (isinf(i0)) {
        return x - .5 * log(2 * M_PI * x);
    }
    return log(i0);
}

double calc_crp_alpha_hyperprior(double alpha)
{
    double logp = 0;
    // invert the effect of log gridding
    // logp += +log(alpha);
    return logp;
}

double calc_continuous_hyperprior(double r, double nu, double s)
{
    double logp = 0;
    // invert the effect of log gridding
    // MAYBE THIS IS THE WRONG PLACE FOR THIS
    // logp += log(r) + log(nu) + log(s);
    return logp;
}

double logaddexp(const vector<double> &logs)
{
    double maximum = *std::max_element(logs.begin(), logs.end());
    double result = 0;
    vector<double>::const_iterator it;
    for (it = logs.begin(); it != logs.end(); it++) {
        result += exp(*it - maximum);
    }
    return log(result) + maximum;
}

// draw_sample_unnormalized(unorm_logps, rand_u)
//
//  unorm_logps is an array [u_0, u_1, ..., u_{n-1}], where each
//  u_i represents a log probability density \log p_i.  rand_u is
//  a dart thrown at the interval [0, 1], i.e. a real number u in
//  [0, 1].  Return the i such that
//
//      \sum_{k=0}^{i-1} p_k <= u*P < \sum_{k=0}^i p_k,
//
//  where P = \sum_{k=0}^{n-1} p_k.
//
//  Strategy: Log/sum/exp and draw_sample_unnormalized.  Let m =
//  \max_j p_j, and let q_i = p_i/m.
//
//  1. Compute M := \max_j u_j, so that
//
//      M = \max_j u_j = \max_j \log p_j
//        = \log \max_j p_j
//        = \log m.
//
//  2. Compute s_i := u_i - M, so that
//
//      s_i = u_i - M = \log p_i - \log m
//          = \log (p_i/m)
//          = \log q_i.
//
//  3. Compute P := \sum_i e^{s_i}, so that
//
//      P = \sum_i e^{s_i} = \sum_i e^{\log q_i} = \sum_i q_i.
//
//  4. Reduce to draw_sample_unnormalized([s_0, s_1, ..., s_{n-1}],
//  \log P, u), where s_i = \log q_i and \log P = \log \sum_i q_i.
//
int draw_sample_unnormalized(const vector<double> &unorm_logps,
    double rand_u)
{
    const size_t N = unorm_logps.size();
    assert(0 < N);
    vector<double> shifted_logps(N);
    double max_el = *std::max_element(unorm_logps.begin(), unorm_logps.end());
    double partition = 0;
    for (size_t i = 0; i < N; i++) {
        shifted_logps[i] = unorm_logps[i] - max_el;
        partition += exp(shifted_logps[i]);
    }
    return draw_sample_with_partition(shifted_logps, log(partition), rand_u);
}

// draw_sample_with_partition(unorm_logps, log_partition, rand_u)
//
//  unorm_logps is an array [u_0, u_1, ..., u_{n-1}], where each
//  u_i represents a log probability density \log p_i.  rand_u is
//  a dart thrown at the interval [0, 1], i.e. a real number u in
//  [0, 1].  log_partition is a real number L representing \log P
//  = \log \sum_j p_j.  Return the i such that:
//
//      \sum_{k=0}^{i-1} p_k <= u*P < \sum_{k=0}^i p_k.
//
//  For each i, let S_i = \sum_{k=0}^{i-1} p_k/P and T_i = u -
//  S_i.  We sequentially compute
//
//      T_0 := u,
//      T_{i+1} := T_i - \exp (u_i - L)
//
//  until the first negative T_{i+1}, since if T_i > 0 > T_{i+1},
//  then
//      u - \sum_{k=0}^{i-1} p_k/P > 0 > u - \sum_{k=0}^i p_k/P,
//  or
//      \sum_{k=0}^{i-1} p_k/P < u < \sum_{k=0}^i p_k/P,
//  hence
//      \sum_{k=0}^{i-1} p_k < u*P < \sum_{k=0}^i p_k.
//
int draw_sample_with_partition(const vector<double> &unorm_logps,
    double log_partition, double rand_u)
{
    const size_t N = unorm_logps.size();
    assert(0 < N);
    for (size_t i = 0; i < N; i++) {
        rand_u -= exp(unorm_logps[i] - log_partition);
        if (rand_u < 0) {
            return i;
        }
    }
    // Since we require rand_u to be in [0, 1] and the partition to be
    // normalized so the p_i sum to P, failing to hit zero by
    // subtracting e^{u_i - \log P} = p_i/P repeatedly can occur only
    // because of numerical error.
    //
    // We hope the error will not be much larger than one machine
    // epsilon away for each operation we do.  Previously this bound
    // was fixed at 1e-10, which should be much larger than we need in
    // most cases.
    //
    // XXX This requires more careful numerical analysis: it is easy
    // to imagine catastrophic cancellation from the subtractions
    // above.
    assert(rand_u < 1000 * N * std::numeric_limits<double>::epsilon());
    return N - 1;
}

// draw_sample_with_partition w/o exp() of ratio and no test for p(last)
// only useful for crp_init or supercluster swapping since no data component
int crp_draw_sample(const vector<int> &counts, int sum_counts, double alpha,
    double rand_u)
{
    int draw = 0;
    double partition = sum_counts + alpha;
    vector<int>::const_iterator it = counts.begin();
    for (; it != counts.end(); it++) {
        rand_u -= (*it / partition);
        if (rand_u < 0) {
            return draw;
        }
        draw++;
    }
    // new cluster
    return draw;
}

// p(alpha | clusters)
double calc_crp_alpha_conditional(const vector<int> &counts,
    double alpha, int sum_counts,
    bool absolute)
{
    int num_clusters = counts.size();
    if (sum_counts == -1) {
        sum_counts = std::accumulate(counts.begin(), counts.end(), 0);
    }
    double logp = lgamma(alpha)         \
        + num_clusters * log(alpha)           \
        - lgamma(alpha + sum_counts);
    // absolute necessary for determining true distribution rather than relative
    if (absolute) {
        double sum_log_gammas = 0;
        vector<int>::const_iterator it = counts.begin();
        for (; it != counts.end(); it++) {
            sum_log_gammas += lgamma(*it);
        }
        logp += sum_log_gammas;
    }
    logp += calc_crp_alpha_hyperprior(alpha);
    return logp;
}

// helper for may calls to calc_crp_alpha_conditional
vector<double> calc_crp_alpha_conditionals(const vector<double> &grid,
    const vector<int> &counts,
    bool absolute)
{
    int sum_counts = std::accumulate(counts.begin(), counts.end(), 0);
    vector<double> logps;
    vector<double>::const_iterator it = grid.begin();
    for (; it != grid.end(); it++) {
        double alpha = *it;
        double logp = calc_crp_alpha_conditional(counts, alpha,
                sum_counts, absolute);
        logps.push_back(logp);
    }
    // note: prior distribution must still be added
    return logps;
}

// p(z=cluster | alpha, clusters)
double calc_cluster_crp_logp(double cluster_weight, double sum_weights,
    double alpha)
{
    if (cluster_weight == 0) {
        cluster_weight = alpha;
    }
    double log_numerator = log(cluster_weight);
    // presumes data has already been removed from the model
    double log_denominator = log(sum_weights + alpha);
    double log_probability = log_numerator - log_denominator;
    return log_probability;
}

void insert_to_continuous_suffstats(int &count,
    double &sum_x, double &sum_x_sq,
    double el)
{
    if (isnan(el)) {
        return;
    }
    count += 1;
    sum_x += el;
    sum_x_sq += el * el;
}

void remove_from_continuous_suffstats(int &count,
    double &sum_x, double &sum_x_sq,
    double el)
{
    if (isnan(el)) {
        return;
    }
    count -= 1;
    sum_x -= el;
    sum_x_sq -= el * el;
}

/*
  r' = r + n
  nu' = nu + n
  m' = m + (X-nm)/(r+n)
  s' = s + C + rm**2 - r'm'**2
*/
// This is equivalent to the updates given in equations 141-144 of [1]
// (NIX Normal hyperparameter updates) with the substitution
//   r = kappa
//   s = nu * sigma^2
// which is also equivalent to the updates in equations 197-200 of
// same (NIG Normal hyperparameter updates), with the substitution
//   r = 1/V
//   nu = 2 * a
//   s = 2 * b
void update_continuous_hypers(int count,
    double sum_x, double sum_x_sq,
    double &r, double &nu,
    double &s, double &mu)
{
    double r_prime = r + count;
    double nu_prime = nu + count;
    double mu_prime = ((r * mu) + sum_x) / r_prime;
    double s_prime = s + sum_x_sq \
        + (r * mu * mu) \
        - (r_prime * mu_prime * mu_prime);
    //
    r = r_prime;
    nu = nu_prime;
    s = s_prime;
    mu = mu_prime;
}

// This is equivalent to the n-subscripted component of equation 170
// of [1] (NIX Normal marginal likelihood) with the same substitution,
// except we can't figure out what the HALF_LOG_2PI term is doing.
// However, we expect that term to cancel in calc_continuous_logp.
double calc_continuous_log_Z(double r, double nu, double s)
{
    double nu_over_2 = .5 * nu;
    double log_Z = nu_over_2 * (LOG_2 - log(s))     \
        + HALF_LOG_2PI                    \
        - .5 * log(r)                 \
        + lgamma(nu_over_2);
    log_Z += calc_continuous_hyperprior(r, nu, s);
    return log_Z;
}

// Assuming log_Z_0 was computed by applying calc_continuous_log_Z to
// the non-updated hyperpriors, this computes equation 170 of [1].
double calc_continuous_logp(int count,
    double r, double nu,
    double s,
    double log_Z_0)
{
    return -count * HALF_LOG_2PI + calc_continuous_log_Z(r, nu, s) - log_Z_0;
}

double calc_continuous_data_logp(int count,
    double sum_x, double sum_x_sq,
    double r, double nu,
    double s, double mu,
    double el,
    double score_0)
{
    if (isnan(el)) {
        return 0;
    }
    insert_to_continuous_suffstats(count, sum_x, sum_x_sq, el);
    update_continuous_hypers(count, sum_x, sum_x_sq, r, nu, s, mu);
    double logp = calc_continuous_logp(count, r, nu, s, score_0);
    return logp;
}

vector<double> calc_continuous_r_conditionals(const vector<double> &r_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double nu,
    double s,
    double mu)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = r_grid.begin(); it != r_grid.end(); it++) {
        double r_prime = *it;
        double nu_prime = nu;
        double s_prime = s;
        double mu_prime = mu;
        double log_Z_0 = calc_continuous_log_Z(r_prime, nu_prime, s_prime);
        update_continuous_hypers(count, sum_x, sum_x_sq,
            r_prime, nu_prime, s_prime, mu_prime);
        double logp = calc_continuous_logp(count,
                r_prime, nu_prime, s_prime,
                log_Z_0);
        // invert the effect of log gridding
        // double prior = log(r_prime);
        // logp += log(prior);
        //
        logps.push_back(logp);
    }
    return logps;
}

vector<double> calc_continuous_nu_conditionals(const vector<double> &nu_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double r,
    double s,
    double mu)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = nu_grid.begin(); it != nu_grid.end(); it++) {
        double r_prime = r;
        double nu_prime = *it;
        double s_prime = s;
        double mu_prime = mu;
        double log_Z_0 = calc_continuous_log_Z(r_prime, nu_prime, s_prime);
        update_continuous_hypers(count, sum_x, sum_x_sq,
            r_prime, nu_prime, s_prime, mu_prime);
        double logp = calc_continuous_logp(count,
                r_prime, nu_prime, s_prime,
                log_Z_0);
        // invert the effect of log gridding
        // double prior = log(nu_prime);
        // logp += log(prior);
        //
        logps.push_back(logp);
    }
    return logps;
}

vector<double> calc_continuous_s_conditionals(const vector<double> &s_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double r,
    double nu,
    double mu)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = s_grid.begin(); it != s_grid.end(); it++) {
        double r_prime = r;
        double nu_prime = nu;
        double s_prime = *it;
        double mu_prime = mu;
        double log_Z_0 = calc_continuous_log_Z(r_prime, nu_prime, s_prime);
        update_continuous_hypers(count, sum_x, sum_x_sq,
            r_prime, nu_prime, s_prime, mu_prime);
        double logp = calc_continuous_logp(count,
                r_prime, nu_prime, s_prime,
                log_Z_0);
        // invert the effect of log gridding
        // double prior = log(s_prime);
        // logp += log(prior);
        //
        // apply gamma prior to s: shape = 2, scale = 10
        // double shape = 2;
        // double scale = 10;
        // logp += -(lgamma(shape) + shape * log(scale)) + (shape-1.)*log(s_prime) - s_prime / scale;
        //
        logps.push_back(logp);
    }
    return logps;
}

vector<double> calc_continuous_mu_conditionals(const vector<double> &mu_grid,
    int count,
    double sum_x,
    double sum_x_sq,
    double r,
    double nu,
    double s)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = mu_grid.begin(); it != mu_grid.end(); it++) {
        double r_prime = r;
        double nu_prime = nu;
        double s_prime = s;
        double mu_prime = *it;
        double log_Z_0 = calc_continuous_log_Z(r_prime, nu_prime, s_prime);
        update_continuous_hypers(count, sum_x, sum_x_sq,
            r_prime, nu_prime, s_prime, mu_prime);
        double logp = calc_continuous_logp(count,
                r_prime, nu_prime, s_prime,
                log_Z_0);
        // apply prior to mu
        // double sigma = 1E4;
        // double mean = 0;
        // boost::math::normal_distribution<> nd(mean, sigma);
        // double prior = boost::math::pdf(nd, mu_prime);
        // double log_prior = log(prior);
        //
        // Does this not work?
        // double log_prior = -log(sigma) - .5 * log(2.0 * M_PI)  + 0.5 * ((mu_prime-mean)*(mu_prime-mean)) / ( sigma*sigma );
        // logp += log_prior;
        logps.push_back(logp);
    }
    return logps;
}

double calc_multinomial_marginal_logp(int count,
    const vector<int> &counts,
    int K,
    double dirichlet_alpha)
{
    double sum_lgammas = 0;
    for (size_t key = 0; key < counts.size(); key++) {
        int label_count = counts[key];
        sum_lgammas += lgamma(label_count + dirichlet_alpha);
    }
    int missing_labels = K - counts.size();
    if (missing_labels != 0) {
        sum_lgammas += missing_labels * lgamma(dirichlet_alpha);
    }
    double marginal_logp = lgamma(K * dirichlet_alpha)  \
        - K * lgamma(dirichlet_alpha)     \
        + sum_lgammas             \
        - lgamma(count + K * dirichlet_alpha);
    return marginal_logp;
}

double calc_multinomial_predictive_logp(double element,
    const vector<int> &counts,
    int sum_counts,
    int K, double dirichlet_alpha)
{
    if (isnan(element)) {
        return 0;
    }
    assert(0 <= element);
    assert(element < K);
    assert(element == trunc(element));
    int i = static_cast<int>(element);
    double numerator = dirichlet_alpha + counts[i];
    double denominator = sum_counts + K * dirichlet_alpha;
    return log(numerator) - log(denominator);
}

vector<double> calc_multinomial_dirichlet_alpha_conditional(
    const vector<double> &dirichlet_alpha_grid,
    int count,
    const vector<int> &counts,
    int K)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = dirichlet_alpha_grid.begin(); it != dirichlet_alpha_grid.end();
        it++) {
        double dirichlet_alpha = *it;
        double logp = calc_multinomial_marginal_logp(count, counts, K,
                dirichlet_alpha);
        logps.push_back(logp);
    }
    return logps;
}


// Cyclic component model
void insert_to_cyclic_suffstats(int &count,
    double &sum_sin_x, double &sum_cos_x,
    double el)
{
    if (isnan(el)) {
        return;
    }
    ++count;
    sum_sin_x += sin(el);
    sum_cos_x += cos(el);
}

void remove_from_cyclic_suffstats(int &count,
    double &sum_sin_x, double &sum_cos_x,
    double el)
{
    if (isnan(el)) {
        return;
    }
    --count;
    sum_sin_x -= sin(el);
    sum_cos_x -= cos(el);
}

void update_cyclic_hypers(int count,
    double sum_sin_x, double sum_cos_x,
    double kappa, double &a, double &b)
{
    double p_cos = kappa * sum_cos_x + a * cos(b);
    double p_sin = kappa * sum_sin_x + a * sin(b);
    double an = sqrt(p_cos * p_cos + p_sin * p_sin);
    double bn =  -atan2(p_cos, p_sin) + M_PI / 2.0;
    //
    a = an;
    b = bn;
}

double calc_cyclic_log_Z(double a)
{
    return log_bessel_0(a);
}

double calc_cyclic_logp(int count, double kappa, double a, double log_Z_0)
{
    double logp = -double(count) * (LOG_2PI + log_bessel_0(kappa));
    logp += calc_cyclic_log_Z(a) - log_Z_0;
    return logp;
}

double calc_cyclic_data_logp(int count,
    double sum_sin_x, double sum_cos_x,
    double kappa, double a, double b,
    double el)
{
    if (isnan(el)) {
        return 0;
    }
    double an, bn, am, bm;
    an = a;
    am = a;
    bn = b;
    bm = b;
    update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa, an, bn);
    update_cyclic_hypers(count + 1, sum_sin_x + sin(el), sum_cos_x + cos(el), kappa,
        am, bm);
    double logp = -LOG_2PI - log_bessel_0(kappa);
    logp += calc_cyclic_log_Z(am) - calc_cyclic_log_Z(an);
    return logp;
}

vector<double> calc_cyclic_a_conditionals(const vector<double> &a_grid,
    int count,
    double sum_sin_x,
    double sum_cos_x,
    double kappa,
    double b)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = a_grid.begin(); it != a_grid.end(); it++) {
        double kappa_prime = kappa;
        double a_prime = *it;
        double b_prime = b;
        double log_Z_0 = calc_cyclic_log_Z(a_prime);
        update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa_prime, a_prime,
            b_prime);
        double logp = calc_cyclic_logp(count, kappa_prime, a_prime, log_Z_0);
        logps.push_back(logp);
    }
    return logps;
}
vector<double> calc_cyclic_b_conditionals(const vector<double> &b_grid,
    int count,
    double sum_sin_x,
    double sum_cos_x,
    double kappa,
    double a)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = b_grid.begin(); it != b_grid.end(); it++) {
        double kappa_prime = kappa;
        double a_prime = a;
        double b_prime = *it;
        double log_Z_0 = calc_cyclic_log_Z(a_prime);
        update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa_prime, a_prime,
            b_prime);
        double logp = calc_cyclic_logp(count, kappa_prime, a_prime, log_Z_0);
        logps.push_back(logp);
    }
    return logps;
}
vector<double> calc_cyclic_kappa_conditionals(const vector<double> &kappa_grid,
    int count,
    double sum_sin_x,
    double sum_cos_x,
    double a,
    double b)
{
    vector<double> logps;
    vector<double>::const_iterator it;
    for (it = kappa_grid.begin(); it != kappa_grid.end(); it++) {
        double kappa_prime = *it;
        double a_prime = a;
        double b_prime = b;
        double log_Z_0 = calc_cyclic_log_Z(a_prime);
        update_cyclic_hypers(count, sum_sin_x, sum_cos_x, kappa_prime, a_prime,
            b_prime);
        double logp = calc_cyclic_logp(count, kappa_prime, a_prime, log_Z_0);
        logps.push_back(logp);
    }
    return logps;
}

} // namespace numerics

// References

// [1] Kevin P. Murphy, "Conjugate Bayesian analysis of the Gaussian
// distribution", murphyk@cs.ubc.ca, last updated October 3, 2007
// http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
