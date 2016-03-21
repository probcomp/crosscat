/*
*   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "RandomNumberGenerator.h"
#include "weakprng.h"

using std::vector;

static const double ALPHA = 0.01;	        // null failure rate
static const size_t NSAMPLES = 100000;

// normal_suffstats(v, mean, sumsqdev)
//
//      Compute sufficient statistics for a normal distribution of n =
//      v.size() samples in v,
//
//              mean := \sum_i v[i]/n,
//              sumsqdev := \sum_i (v[i] - mean)^2.
//
static void normal_suffstats(const vector<double> &v,
        double &mean, double &sumsqdev) {
    const size_t n = v.size();
    double m, s;
    size_t i;

    m = s = 0;
    for (i = 0; i < n; i++) {
	const double d = v[i] - m;
	m += d/n;
	s += d*(v[i] - m);
    }

    mean = m;
    sumsqdev = s;
}

// stdnormal_cdf(x)
//
//      Cumulative distribution function of standard normal
//      distribution.
//
static double stdnormal_cdf(double x) {
    return (1 + erf(x/sqrt(2)))/2;
}

// chisquare2_cdf(x)
//
//      Cumulative distribution function of chi-squared distribution
//      with two degrees of freedom.
//
static double chisquare2_cdf(double x) {
    return 1 - exp(-x/2);
}

// normalized_upper_gamma(n, x)
//
//      Normalized incomplete upper gamma integral Q(n, x), for
//      positive integer n:
//
//              Q(n, x) = Gamma(n, x)/Gamma(n)
//              Gamma(n, x) = \int_x^\infty t^{n - 1} e^{-t} dt
//              Gamma(n) = (n - 1)!
//
//      We use identity (8.4.10) from NIST DLMF 8.4 (Incomplete Gamma
//      and Related Functions, Incomplete Gamma Functions, Special
//      Values) <http://dlmf.nist.gov/8.4.E10>:
//
//              Q(n + 1, x) = e^{-x} \sum_{k = 0}^n x^k/k!
//
//      [1] NIST Digital Library of Mathematical Functions,
//      <http://dlmf.nist.gov/>, Release 1.0.10 of 2015-08-07.
//
static double normalized_upper_gamma(unsigned n, double x) {
    double e;
    unsigned k;

    // Q(n + 1, x) = e^{-x} \sum_{k = 0}^n x^k/k!
    // Q((n - 1) + 1, x) = e^{-x} \sum_{k = 0}^{n - 1} x^k/k!
    // Q(n, x) = e^{-x} \sum_{k = 0}^{n - 1} x^k/k!
    e = 0;
    for (k = 0; k < n; k++) {
        double f = 1;
        unsigned i;

        // f := x^k/k!
        for (i = 0; i < k; i++)
            f *= x/(i + 1);
        e += f;
    }

    return exp(-x)*e;
}

// chisquare8_cdf(x)
//
//      CDF for chi^2_8.
//
static double chisquare8_cdf(double x) {
    return 1 - normalized_upper_gamma(8/2, x/2);
}

// t2_cdf(x)
//
//      Cumulative distribution function of Student t distribution
//      with two degrees of freedom.
//
static double t2_cdf(double x) {
    return 0.5 + x/(2*sqrt(2 + x*x));
}

// Psi-test for goodness of fit -- scaled KL divergence of the
// theoretical distribution from the empirical distribution:
//
//                            O_i
//      psi = 2 \sum O_i log -----
//                i           E_i
//
// where O_i is the observed count for the ith bin and E_i is the
// expected count for the ith bin, equal to N*Pr[Bin = i].
//
// Psi is scaled so that under the null hypothesis that the two are
// equal, the distribution of the test statistic psi converges to the
// chi^2 distribution as the number of samples grows without bound.
//
// Chi^2 critical value at significance level 0.01 with 100 degrees of
// freedom derived from
//
//      NIST/SEMATECH e-Handbook of Statistical Methods, Section
//      1.3.6.7.4 `Critical Values of the Chi-Square Distribution',
//      <http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm>,
//      retrieved 2016-03-14.

static const size_t PSI_DF = 100;               // degrees of freedom
static const double PSI_CRITICAL = 135.807;     // critical value

static bool psi_test(const vector<size_t> &counts,
        const vector<double> &probabilities, size_t nsamples) {
    size_t i;
    double psi = 0;

    assert(PSI_DF == counts.size());
    assert(PSI_DF == probabilities.size());

    for (i = 0; i < PSI_DF; i++) {
        // We treat empty bins as zero because we are evaluating
        //
        //      psi = 2 \sum_i O_i log O_i/E_i
        //          = 2 \sum_i N*(O_i/N) log (N*(O_i)/N / N Pr[Bin = i])
        //          = 2 N \sum_i (O_i/N) log (O_i/N)/Pr[Bin = i]
        //          = 2 N \sum_i f_i log f_i/p_i
        //          = 2 N \sum_i f_i log f_i + f_i log p_i.
        //
        // where f_i = O_i/N and p_i = Pr[Bin = i].  As f_i ---> 0, so
        // does f_i log f_i.
        if (counts[i] == 0)
            continue;
        psi += counts[i] * log(counts[i] / (nsamples*probabilities[i]));
    }
    psi *= 2;

    return psi <= PSI_CRITICAL;
}

// Shapiro-Wilk test for normality, described in
//
//      S.S. Shapiro and M.B. Wilk, `An analysis of variance test for
//      normality (complete samples)', Biometrika 52(3,4), December
//      1965, pp. 591--611.
//
//      Algorithm and statistic tables summarized in Section 3
//      `Summary of operational information' on p. 602.

static const size_t SHAPIRO_WILK_DF = 50;
static const double SHAPIRO_WILK_CRITICAL = .930;
static const double SHAPIRO_WILK_A[SHAPIRO_WILK_DF/2] = {
    .3751, .2574, .2260, .2032, .1847,
    .1691, .1554, .1430, .1317, .1212,
    .1113, .1020, .0932, .0846, .0764,
    .0685, .0608, .0532, .0459, .0386,
    .0314, .0244, .0174, .0104, .0035,
};

static bool shapiro_wilk_test(const vector<double> &x) {
    const size_t n = SHAPIRO_WILK_DF;
    vector<double> y = x;
    double mean, sumsqdev;
    double b, W;
    size_t i;

    assert(x.size() == n);

    // Compute sample mean and sum of squared deviations.
    normal_suffstats(x, mean, sumsqdev);

    // Compute order statistics.
    std::sort(y.begin(), y.end());

    // Compute b.
    b = 0;
    for (i = 0; i < n/2; i++)
        b += SHAPIRO_WILK_A[i]*(y[n - i - 1] - y[i]);

    // Compute W test statistic, W = b^2/S^2.
    W = b*b/sumsqdev;

    // Return test result.  Alternative hypotheses yield lower values.
    return W >= SHAPIRO_WILK_CRITICAL;
}

// random_seed()
//
//      Return a nondeterministic choice of seed.
static int random_seed(void) {
    std::ifstream devurandom("/dev/urandom");
    int seed;

    assert(devurandom.good());
    devurandom.read(reinterpret_cast<char *>(&seed), sizeof(seed));
    assert(devurandom.good());

    return seed;
}

static void test_uniform_integer(RandomNumberGenerator &rng) {
    vector<size_t> counts(PSI_DF);
    vector<double> probabilities(PSI_DF);
    size_t i;

    for (i = 0; i < probabilities.size(); i++)
        probabilities[i] = 1/static_cast<double>(PSI_DF);
    for (i = 0; i < NSAMPLES; i++)
        counts[rng.nexti(PSI_DF)]++;
    assert(psi_test(counts, probabilities, NSAMPLES));

    // Check that the psi test has sufficient statistical power to
    // detect the modulo bias.
    std::fill(counts.begin(), counts.end(), 0);
    for (i = 0; i < NSAMPLES; i++)
        counts[rng.nexti(2*PSI_DF + 1) % PSI_DF]++;
    assert(!psi_test(counts, probabilities, NSAMPLES));
}

static void test_uniform01(RandomNumberGenerator &rng) {
    vector<size_t> counts(PSI_DF);
    vector<double> probabilities(PSI_DF);
    size_t i;

    for (i = 0; i < probabilities.size(); i++)
        probabilities[i] = 1/static_cast<double>(PSI_DF);
    for (i = 0; i < NSAMPLES; i++)
        counts[static_cast<size_t>(floor(rng.next()*PSI_DF))]++;
    assert(psi_test(counts, probabilities, NSAMPLES));
}

static void test_stdnormal_sw(RandomNumberGenerator &rng) {
    vector<double> samples(SHAPIRO_WILK_DF);
    size_t i;

    for (i = 0; i < samples.size(); i++)
	samples[i] = rng.stdnormal();
    assert(shapiro_wilk_test(samples));
}

class sampler {
public:
    virtual double operator()(RandomNumberGenerator &rng) const = 0;
};

class stdnormal_sampler : public sampler {
public:
    virtual double operator()(RandomNumberGenerator &rng) const {
        return rng.stdnormal();
    }
};

class chisquare_sampler : public sampler {
public:
    explicit chisquare_sampler(double nu) : _nu(nu) {}
    virtual double operator()(RandomNumberGenerator &rng) const {
        return rng.chisquare(_nu);
    }
private:
    double _nu;
};

class t_sampler : public sampler {
public:
    explicit t_sampler(double nu) : _nu(nu) {}
    virtual double operator()(RandomNumberGenerator &rng) const {
        return rng.student_t(_nu);
    }
private:
    double _nu;
};

static void cdf_bins(double (*F)(double), double lo, double hi,
        vector<double> &probabilities) {
    const double nbins = static_cast<double>(probabilities.size() - 2);
    const double w = (hi - lo)/nbins;
    double x0, x1;
    size_t i;

    x0 = -HUGE_VAL;
    x1 = lo;
    probabilities[0] = F(x1) - 0;

    for (i = 1; i < probabilities.size() - 1; i++) {
        x0 = lo + (i - 1)*w;
        x1 = lo + i*w;
        probabilities[i] = F(x1) - F(x0);
    }

    x0 = hi;
    x1 = HUGE_VAL;
    probabilities[probabilities.size() - 1] = 1 - F(x0);
}

static void sample_bins(const sampler &sample, double lo, double hi,
        RandomNumberGenerator &rng, vector<size_t> &counts) {
    const double nbins = static_cast<double>(counts.size() - 2);
    const double w = (hi - lo)/nbins;
    double x;
    size_t i;

    for (i = 0; i < NSAMPLES; i++) {
        x = sample(rng);
        if (x < lo)
            counts[0]++;
        else if (hi <= x)
            counts[counts.size() - 1]++;
        else
            counts.at(1 + static_cast<size_t>(floor((x - lo)/w)))++;
    }
}

static void test_stdnormal_psi(RandomNumberGenerator &rng) {
    vector<size_t> counts(PSI_DF);
    vector<double> probabilities(PSI_DF);
    const double lo = -5;
    const double hi = +5;

    cdf_bins(stdnormal_cdf, lo, hi, probabilities);

    sample_bins(stdnormal_sampler(), lo, hi, rng, counts);
    assert(psi_test(counts, probabilities, NSAMPLES));

    // Check that the psi test has sufficient statistical power to
    // distinguish a normal from a low-degree Student t.
    std::fill(counts.begin(), counts.end(), 0);
    sample_bins(t_sampler(10), lo, hi, rng, counts);
    assert(!psi_test(counts, probabilities, NSAMPLES));
}

static void test_chisquare_psi(RandomNumberGenerator &rng) {
    vector<size_t> counts(PSI_DF);
    vector<double> probabilities(PSI_DF);
    const double lo = 0.1;
    const double hi = 10;

    cdf_bins(chisquare2_cdf, lo, hi, probabilities);
    sample_bins(chisquare_sampler(2), lo, hi, rng, counts);
    assert(psi_test(counts, probabilities, NSAMPLES));

    std::fill(counts.begin(), counts.end(), 0);
    cdf_bins(chisquare8_cdf, lo, hi, probabilities);
    sample_bins(chisquare_sampler(8), lo, hi, rng, counts);
    assert(psi_test(counts, probabilities, NSAMPLES));
}

static void test_student_t_psi(RandomNumberGenerator &rng) {
    vector<size_t> counts(PSI_DF);
    vector<double> probabilities(PSI_DF);
    const double lo = -10;
    const double hi = +10;

    cdf_bins(t2_cdf, lo, hi, probabilities);
    sample_bins(t_sampler(2), lo, hi, rng, counts);
    assert(psi_test(counts, probabilities, NSAMPLES));
}

int main(int argc, char **argv) {
    std::cout << __FILE__ << "..." << std::endl;

    if (crypto_weakprng_selftest() != 0)
        assert(!"crypto selftest failed");

    RandomNumberGenerator rng(random_seed());
    test_uniform_integer(rng);
    test_uniform01(rng);
    test_stdnormal_sw(rng);
    test_stdnormal_psi(rng);
    test_chisquare_psi(rng);
    test_student_t_psi(rng);

    std::cout << __FILE__ << " passed" << std::endl;
}
