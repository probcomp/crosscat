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

#define __STDC_CONSTANT_MACROS  // Make <stdint.h> define UINT64_C &c.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdint.h>

#include "RandomNumberGenerator.h"

static inline unsigned bitcount64(uint64_t x)
{
    // Count two-bit groups.
    x -= ((x >> 1) & UINT64_C(0x5555555555555555));
    // Add to four-bit groups, carefully masking carries and garbage.
    x = ((x >> 2) & UINT64_C(0x3333333333333333)) +
        (x & UINT64_C(0x3333333333333333));
    // Add to eight-bit groups.
    x = (((x >> 4) + x) & UINT64_C(0x0f0f0f0f0f0f0f0f));
    // Add all eight-bit groups in the leftmost column of a multiply.
    // Bit counts are small enough no other columns will carry.
    return (x * UINT64_C(0x0101010101010101)) >> 56;
}

static inline unsigned clz64(uint64_t x)
{
    // Round up to a power of two minus one.
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    // Count the bits thus set, and subtract from 64 to count leading
    // zeros.
    return 64 - bitcount64(x);
}

//////////////////////////////////
// return a random real between
// 0 and 1 with uniform dist
double RandomNumberGenerator::next()
{
    int e = -64;
    uint64_t s;
    unsigned d;
    // Draw a significand from an infinite stream of bits.
    while ((s = crypto_weakprng_64(&_weakprng)) == 0) {
        e -= 64;
        // Just return zero if the exponent is so small there are no
        // floating-point numbers that tiny.
        if (e < -1074) {
            return 0;
        }
    }
    // Shift leading zeros into the exponent and fill trailing bits of
    // significand uniformly at random.
    if ((d = clz64(s)) != 0) {
        e -= d;
        s <<= d;
        s |= crypto_weakprng_64(&_weakprng) >> (64 - d);
    }
    // Set sticky bit, since there is (almost surely) another 1 in the
    // bit stream, to avoid a bias toward `even'.
    s |= 1;
    // Return s * 2^e.
    return ldexp(static_cast<double>(s), e);
}

//////////////////////////////
// return a random int bewteen
// zero and max - 1 with uniform
// dist if called with same max
int RandomNumberGenerator::nexti(int bound)
{
    assert(0 < bound);
    return crypto_weakprng_below(&_weakprng, bound);
}

/////////////////////////////
// standard normal samples
// via Box-Muller transform
double RandomNumberGenerator::stdnormal()
{
    double u, v;
    u = next();
    v = next();
    return sqrt(-2 * log(u)) * sin(2 * M_PI * v);
}

/////////////////////////////
// standard Gamma samples
//
//       George Marsaglia & Wai Wan Tsang, `A simple method for
//       generating gamma variables', ACM Transactions on Mathematical
//       Software 26(3), September 2000.  DOI: 10.1145/358407.358414
//       URI: https://dl.acm.org/citation.cfm?doid=358407.358414
double RandomNumberGenerator::stdgamma(double alpha)
{
    const double d = alpha - (double)1 / 3;
    const double c = 1 / sqrt(9 * d);
    double x, u, v;
    // The clients currently do not need alpha < 1.  Should they, the
    // reference contains a note (at the end of Section 6) on how to
    // boost the alpha parameter:
    //   stdgamma(alpha) = stdgamma(alpha+1) * (uniform() ** (1/alpha))
    assert(1 <= alpha);
    for (;;) {
        x = stdnormal();
        v = 1 + x * c;
        if (v <= 0) {
            continue;
        }
        v = v * v * v;
        u = next();
        if (u < 1 - 0.0331 * ((x * x) * (x * x)) ||
            log(u) < x * x / 2 + d - d * v + d * log(v)) {
            return d * v;
        }
    }
}

/////////////////////////////
// chi^2 samples
double RandomNumberGenerator::chisquare(double nu)
{
    double shape = nu / 2;
    double scale = 2;
    return scale * stdgamma(shape);
}

/////////////////////////////
// Student's t-distribution samples
double RandomNumberGenerator::student_t(double nu)
{
    return stdnormal() * sqrt(nu / chisquare(nu));
}

/////////////////////////////
// control the seed
void RandomNumberGenerator::set_seed(std::time_t seed)
{
    uint8_t seedbuf[crypto_weakprng_SEEDBYTES] = {0};
    size_t i;
    for (i = 0; i < std::min(sizeof seedbuf, sizeof seed); i++) {
        seedbuf[i] = seed >> (8 * i);
    }
    crypto_weakprng_seed(&_weakprng, seedbuf);
}
