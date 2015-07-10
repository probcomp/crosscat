/*
*   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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

#include <limits>

#include "numerics.h"
#include "utils.h"

using namespace std;

static void test_draw(void) {
    using numerics::draw_sample_unnormalized;
    const double epsilon = std::numeric_limits<double>::epsilon();
    const double denorm_min = std::numeric_limits<double>::denorm_min();

    vector<double> weights;

    weights.resize(4);
    weights[0] = log(1);
    weights[1] = log(2);
    weights[2] = log(4);
    weights[3] = log(8);
    assert(draw_sample_unnormalized(weights, 0.0) == 0);
    assert(draw_sample_unnormalized(weights, 0.1) == 1);
    assert(draw_sample_unnormalized(weights, 0.2) == 2);
    assert(draw_sample_unnormalized(weights, 0.3) == 2);
    assert(draw_sample_unnormalized(weights, 0.4) == 2);
    assert(draw_sample_unnormalized(weights, 0.5) == 3);
    assert(draw_sample_unnormalized(weights, 0.6) == 3);
    assert(draw_sample_unnormalized(weights, 0.7) == 3);
    assert(draw_sample_unnormalized(weights, 0.8) == 3);
    assert(draw_sample_unnormalized(weights, 0.9) == 3);
    assert(draw_sample_unnormalized(weights, 1 - epsilon/2) == 3);

    weights.resize(9);
    weights[0] = -denorm_min;
    for (size_t i = 1; i < weights.size(); i++)
	weights[i] = -1;
    assert(draw_sample_unnormalized(weights, 0) == 0);
    assert(draw_sample_unnormalized(weights, 1 - epsilon/2) < weights.size());
}

static void test_linspace(void) {
    vector<double> v = linspace(0, 1, 7);
    assert(v[6] == 1);
}

static void test_logaddexp(void) {
    vector<double> v(1);

    v[0] = 1000;
    assert(numerics::logaddexp(v) == 1000);
    v[0] = -1000;
    assert(numerics::logaddexp(v) == -1000);
}

int main(int argc, char** argv) {
    test_draw();
    test_linspace();
    test_logaddexp();

    return 0;
}
