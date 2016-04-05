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

// Program to check the modified Bessel functions of the first (I_0)
// and second (I_1) kinds against boost at points sampled randomly
// from N(0, 1), N(15, 1), and U[-709, +709].  The edge cases are
// zero, since I_0 is even and I_1 is odd; 15, where we switch from
// one rational approximation to another; and +/- 709, which are about
// where the values overflow the representable range of IEEE 754
// double-precision floating-point numbers.
//
// Usage:
//
//      c++ -I../include/CrossCat -o bessamp bessamp.cpp ../src/RandomNumberGenerator.cpp ../src/numerics.cpp ../src/weakprng.cpp
//      ./bessamp | head -n <nsamples>
//
// The output is
//
//      nu x boost local relerr
//
// where nu \in {0, 1}, x \in [-709, +709], boost is I_nu(x) evaluated
// using boost, local is I_nu(x) evaluated using the local numerics
// code, and relerr is the relative error of local from boost, or the
// absolute magnitude of local if boost = 0.
//
// The invocation
//
//      ./bessamp | awk '$5 > 1e-14'
//
// will show points at which we disagree with boost by more than one
// digit.  So far I have not seen it show any.

#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <cstdio>
#include <limits>

#include "RandomNumberGenerator.h"
#include "numerics.h"

static double relerr(double expected, double actual) {
    return fabs(expected == 0 ? actual : (actual - expected)/actual);
}

static double check(unsigned nu, double x) {
    double expected, actual, error;

    expected = boost::math::cyl_bessel_i(nu, x);
    actual = nu == 0 ? numerics::i_0(x) : numerics::i_1(x);
    error = relerr(expected, actual);
    if (printf("%u %.17e %.17e %.17e %.17e\n", nu, x, expected, actual, error)
        < 0)
        abort();
}

int main(int argc, char **argv) {
    const double epsilon = std::numeric_limits<double>::epsilon();
    RandomNumberGenerator rng;
    unsigned nu;

    for (;;) {
        for (nu = 0; nu < 2; nu++) {
            check(nu, rng.stdnormal());
            check(nu, 15 + rng.stdnormal());
            check(nu, rng.nexti(+709 - -709) + -709);
        }
    }
}
