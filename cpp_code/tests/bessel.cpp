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

// Program to generate test values for the modified Bessel function of
// the first kind using boost.  Usage:
//
//      c++ -o bessel bessel.cpp
//      ./bessel

#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <cstdio>

int main(int argc, char **argv) {
    const size_t n = 100;
    const double lo = -709;
    const double hi = +709;
    const double w = (hi - lo)/n;
    unsigned nu;
    size_t i;

    (void)argc;
    (void)argv;

    for (nu = 0; nu < 2; nu++) {
        if (printf("static const double i%ue[][2] = {\n", nu) < 0) {
            perror("printf");
            return 1;
        }
        if (printf("    // Uniform [-709, +709] grid\n") < 0) {
            perror("printf");
            return 1;
        }
        for (i = 0; i < n; i++) {
            const double x = lo + i*w;
            const double y = boost::math::cyl_bessel_i(nu, x);
            if (printf("    { %.17e, %.17e },\n", x, y) < 0) {
                perror("printf");
                return 1;
            }
        }
        if (printf("    // N(0, 1e-4) grid\n") < 0) {
            perror("printf");
            return 1;
        }
        for (i = 0; i < n; i++) {
            boost::math::normal norm(0, 1e-4);
            const double p = static_cast<double>(i + 1)/(n + 2);
            const double x = boost::math::quantile(norm, p);
            const double y = boost::math::cyl_bessel_i(nu, x);
            if (printf("    { %.17e, %.17e },\n", x, y) < 0) {
                perror("printf");
                return 1;
            }
        }
        if (printf("    // N(15, 1e-4) grid\n") < 0) {
            perror("printf");
            return 1;
        }
        for (i = 0; i < n; i++) {
            boost::math::normal norm(15, 1e-4);
            const double p = static_cast<double>(i + 1)/(n + 2);
            const double x = boost::math::quantile(norm, p);
            const double y = boost::math::cyl_bessel_i(nu, x);
            if (printf("    { %.17e, %.17e },\n", x, y) < 0) {
                perror("printf");
                return 1;
            }
        }
        if (printf("};\n") < 0) {
            perror("printf");
            return 1;
        }
    }

    return 0;
}
