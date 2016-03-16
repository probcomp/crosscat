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

// Psi-test for goodness of fit -- scaled KL divergence of the
// theoretical distribution from the empirical distribution:
//
//                            O_i
//      psi = 2 \sum O_i log -----
//                i           E_i
//
// where o[i] is the observed count for the ith bin and e[i] is the
// expected count for the ith bin.
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

    for (i = 0; i < PSI_DF; i++)
        psi += counts[i] * log(counts[i] / (nsamples*probabilities[i]));
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

static void test_stdnormal(RandomNumberGenerator &rng) {
    vector<double> samples(SHAPIRO_WILK_DF);
    size_t i;

    for (i = 0; i < samples.size(); i++)
	samples[i] = rng.stdnormal();
    assert(shapiro_wilk_test(samples));
}

int main(int argc, char **argv) {
    std::cout << __FILE__ << "..." << std::endl;

    if (crypto_weakprng_selftest() != 0)
        assert(!"crypto selftest failed");

    RandomNumberGenerator rng(random_seed());
    test_uniform_integer(rng);
    test_uniform01(rng);
    test_stdnormal(rng);

    std::cout << __FILE__ << " passed" << std::endl;
}
