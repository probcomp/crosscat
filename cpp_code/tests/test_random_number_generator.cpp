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

static const size_t NSAMPLES = 100000;

// The tests are calibrated to have significance level alpha = 0.01,
// i.e. the rejection rate under the null hypothesis, or false
// rejection rate.  If there are n tests, then the false rejection
// rate of the entire suite is 1 - Binom(0; n, alpha).  For n = 10,
// this is about 10%, i.e. about one in ten runs of the test suite
// would fail even if the code is correct.
//
// To reduce the false rejection rate, we could change alpha, but that
// requires changing all the critical values in all the tests.
// Critical values for smaller values of alpha are not always easy to
// find and audit, e.g. the critical value for the smallest alpha
// reported in the Shapiro-Wilk paper is 0.01.
//
// An easy alternative way to reduce the false rejection rate, at the
// expense of statistical power, is to repeat each test in k
// independent trials and reject only if it failed all k.  It is then
// as if the significance level were Binom(k; k, alpha).  For k = 2,
// this is approximately alpha^2.  For alpha = 0.01, the resulting
// false rejection rate of the entire suite for n = 10 tests is then
// approximately 1 - Binom(0; 10, alpha^2) or about 0.1%.  If we wrote
// 90 more tests, then the false rejection rate would still be only
// about 1% for the whole test suite.
//
// We could also stop early as soon as we cross the threshold of
// required passes, without changing the false rejection rate, but
// that requires writing more code.

static const unsigned NPASSES_MIN = 1;
static const unsigned NTRIALS = 2;

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

// stdgamma11_cdf(x)
//
//      CDF for Gamma(alpha = 11).
//
static double stdgamma11_cdf(double x) {
    return 1 - normalized_upper_gamma(11, x);
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
static const double PSI_CRITICAL = 135.807;     // critical value, alpha = .01

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
static const double SHAPIRO_WILK_CRITICAL = .930;  // alpha = .01
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
    vector<double> probabilities(PSI_DF);
    size_t i;
    unsigned trial, passes;

    for (i = 0; i < probabilities.size(); i++)
        probabilities[i] = 1/static_cast<double>(PSI_DF);

    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        for (i = 0; i < NSAMPLES; i++)
            counts[rng.nexti(PSI_DF)]++;
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);

    // Check that the psi test has sufficient statistical power to
    // detect the modulo bias.
    vector<size_t> counts(PSI_DF);
    for (i = 0; i < NSAMPLES; i++)
      counts[rng.nexti(2*PSI_DF + 1) % PSI_DF]++;
    assert(!psi_test(counts, probabilities, NSAMPLES));
}

static void test_uniform01(RandomNumberGenerator &rng) {
    vector<double> probabilities(PSI_DF);
    size_t i;
    unsigned trial, passes;

    for (i = 0; i < probabilities.size(); i++)
        probabilities[i] = 1/static_cast<double>(PSI_DF);

    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        for (i = 0; i < NSAMPLES; i++)
            counts[static_cast<size_t>(floor(rng.next()*PSI_DF))]++;
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);
}

static void test_stdnormal_sw(RandomNumberGenerator &rng) {
    unsigned trial, passes;

    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<double> samples(SHAPIRO_WILK_DF);
        size_t i;

        for (i = 0; i < samples.size(); i++)
            samples[i] = rng.stdnormal();
        passes += shapiro_wilk_test(samples);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);
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

class stdgamma_sampler : public sampler {
public:
    explicit stdgamma_sampler(double alpha) : _alpha(alpha) {}
    virtual double operator()(RandomNumberGenerator &rng) const {
        return rng.stdgamma(_alpha);
    }
private:
    double _alpha;
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
    vector<double> probabilities(PSI_DF);
    const double lo = -5;
    const double hi = +5;
    unsigned trial, passes;

    cdf_bins(stdnormal_cdf, lo, hi, probabilities);

    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        sample_bins(stdnormal_sampler(), lo, hi, rng, counts);
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);

    // Check that the psi test has sufficient statistical power to
    // distinguish a normal from a low-degree Student t.
    vector<size_t> counts(PSI_DF);
    sample_bins(t_sampler(10), lo, hi, rng, counts);
    assert(!psi_test(counts, probabilities, NSAMPLES));
}

static void test_stdgamma_psi(RandomNumberGenerator &rng) {
    vector<double> probabilities(PSI_DF);
    const double lo = 0.1;
    const double hi = 20;
    unsigned trial, passes;

    cdf_bins(stdgamma11_cdf, lo, hi, probabilities);

    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        sample_bins(stdgamma_sampler(11), lo, hi, rng, counts);
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);
}

static void test_chisquare_psi(RandomNumberGenerator &rng) {
    vector<double> probabilities(PSI_DF);
    const double lo = 0.1;
    const double hi = 10;
    unsigned trial, passes;

    cdf_bins(chisquare2_cdf, lo, hi, probabilities);
    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        sample_bins(chisquare_sampler(2), lo, hi, rng, counts);
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);

    cdf_bins(chisquare8_cdf, lo, hi, probabilities);
    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        sample_bins(chisquare_sampler(8), lo, hi, rng, counts);
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);
}

static void test_student_t_psi(RandomNumberGenerator &rng) {
    vector<double> probabilities(PSI_DF);
    const double lo = -10;
    const double hi = +10;
    unsigned trial, passes;

    cdf_bins(t2_cdf, lo, hi, probabilities);
    passes = 0;
    for (trial = 0; trial < NTRIALS; trial++) {
        vector<size_t> counts(PSI_DF);

        sample_bins(t_sampler(2), lo, hi, rng, counts);
        passes += psi_test(counts, probabilities, NSAMPLES);
        if (passes >= NPASSES_MIN)
            break;
    }
    assert(passes >= NPASSES_MIN);
}

static uint32_t le32dec(const uint8_t *p) {
    uint32_t v = 0;

    v |= (uint32_t)*p++ << 0;
    v |= (uint32_t)*p++ << 8;
    v |= (uint32_t)*p++ << 16;
    v |= (uint32_t)*p++ << 24;

    return v;
}

static void test_weakprng(void) {
    static const struct {
        uint8_t key[32];
        uint8_t block[2][64];
    } test[] = {
        // Test vectors from J. Strombergson, `Test Vectors for the
        // Stream Cipher ChaCha', Internet-Draft, December 2013,
        // <https://tools.ietf.org/html/draft-strombergson-chacha-test-vectors-01>.
        //
        // We adopt only those tests of ChaCha8 with a 256-bit key and
        // all-zero IV, because that's what weakprng computes.
        {
            {                   // key
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            },
            {                   // block
                {               // [0]
                    0x3e,0x00,0xef,0x2f,0x89,0x5f,0x40,0xd6,
                    0x7f,0x5b,0xb8,0xe8,0x1f,0x09,0xa5,0xa1,
                    0x2c,0x84,0x0e,0xc3,0xce,0x9a,0x7f,0x3b,
                    0x18,0x1b,0xe1,0x88,0xef,0x71,0x1a,0x1e,
                    0x98,0x4c,0xe1,0x72,0xb9,0x21,0x6f,0x41,
                    0x9f,0x44,0x53,0x67,0x45,0x6d,0x56,0x19,
                    0x31,0x4a,0x42,0xa3,0xda,0x86,0xb0,0x01,
                    0x38,0x7b,0xfd,0xb8,0x0e,0x0c,0xfe,0x42,
                },
                {               // [1]
                    0xd2,0xae,0xfa,0x0d,0xea,0xa5,0xc1,0x51,
                    0xbf,0x0a,0xdb,0x6c,0x01,0xf2,0xa5,0xad,
                    0xc0,0xfd,0x58,0x12,0x59,0xf9,0xa2,0xaa,
                    0xdc,0xf2,0x0f,0x8f,0xd5,0x66,0xa2,0x6b,
                    0x50,0x32,0xec,0x38,0xbb,0xc5,0xda,0x98,
                    0xee,0x0c,0x6f,0x56,0x8b,0x87,0x2a,0x65,
                    0xa0,0x8a,0xbf,0x25,0x1d,0xeb,0x21,0xbb,
                    0x4b,0x56,0xe5,0xd8,0x82,0x1e,0x68,0xaa,
                },
            },
        },
        {
            {                   // key
                0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            },
            {                   // block
                {               // [0]
                    0xcf,0x5e,0xe9,0xa0,0x49,0x4a,0xa9,0x61,
                    0x3e,0x05,0xd5,0xed,0x72,0x5b,0x80,0x4b,
                    0x12,0xf4,0xa4,0x65,0xee,0x63,0x5a,0xcc,
                    0x3a,0x31,0x1d,0xe8,0x74,0x04,0x89,0xea,
                    0x28,0x9d,0x04,0xf4,0x3c,0x75,0x18,0xdb,
                    0x56,0xeb,0x44,0x33,0xe4,0x98,0xa1,0x23,
                    0x8c,0xd8,0x46,0x4d,0x37,0x63,0xdd,0xbb,
                    0x92,0x22,0xee,0x3b,0xd8,0xfa,0xe3,0xc8,
                },
                {               // [1]
                    0xb4,0x35,0x5a,0x7d,0x93,0xdd,0x88,0x67,
                    0x08,0x9e,0xe6,0x43,0x55,0x8b,0x95,0x75,
                    0x4e,0xfa,0x2b,0xd1,0xa8,0xa1,0xe2,0xd7,
                    0x5b,0xcd,0xb3,0x20,0x15,0x54,0x26,0x38,
                    0x29,0x19,0x41,0xfe,0xb4,0x99,0x65,0x58,
                    0x7c,0x4f,0xdf,0xe2,0x19,0xcf,0x0e,0xc1,
                    0x32,0xa6,0xcd,0x4d,0xc0,0x67,0x39,0x2e,
                    0x67,0x98,0x2f,0xe5,0x32,0x78,0xc0,0xb4,
                },
            },
        },
    };
    struct crypto_weakprng weakprng;
    unsigned i, j, k;

    if (crypto_weakprng_selftest() != 0)
        assert(!"crypto selftest failed");
    for (i = 0; i < sizeof(test)/sizeof(test[0]); i++) {
        crypto_weakprng_seed(&weakprng, test[i].key);
        for (j = 0; j < 2; j++) {
            for (k = 16; k --> 0;) {
                const uint32_t expected = le32dec(&test[i].block[j][4*k]);
                const uint32_t actual = crypto_weakprng_32(&weakprng);

                assert(expected == actual);
            }
        }
    }
}

int main(int argc, char **argv) {
    std::cout << __FILE__ << "..." << std::endl;

    test_weakprng();

    int seed = (argc == 2? atoi(argv[1]) : random_seed());
    std::cout << "seed " << seed << std::endl;

    RandomNumberGenerator rng(seed);
    test_uniform_integer(rng);
    test_uniform01(rng);
    test_stdnormal_sw(rng);
    test_stdnormal_psi(rng);
    test_stdgamma_psi(rng);
    test_chisquare_psi(rng);
    test_student_t_psi(rng);

    std::cout << __FILE__ << " passed" << std::endl;
}
