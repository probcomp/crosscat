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

using namespace std;

static void test_logaddexp(void) {
    vector<double> v(1);

    v[0] = 1000;
    assert(numerics::logaddexp(v) == 1000);
    v[0] = -1000;
    assert(numerics::logaddexp(v) == -1000);
}

int main(int argc, char** argv) {
    test_logaddexp();

    return 0;
}
