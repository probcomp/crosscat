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

#include <assert.h>
#include <algorithm>

#include "Matrix.h"

int main(int argc, char **argv) {
    const size_t n = 42, m = 87;
    matrix<std::pair<size_t, size_t> > M(n, m);
    size_t i, j;

    // Confirm sizes make sense.
    assert(M.size1() == n);
    assert(M.size2() == m);

    // Confirm storing values works.
    for (i = 0; i < n; i++)
	for (j = 0; j < m; j++)
	    M(i, j) = std::make_pair<size_t, size_t>(i, j);

    // Confirm retrieving values in another order works.
    for (i = n; 0 < i--;) {
	for (j = m; 0 < j--;) {
	    assert(M(i, j).first == i);
	    assert(M(i, j).second == j);
	}
    }

    // Confirm copying works.
    matrix<std::pair<size_t, size_t> > N = M;
    assert(N.size1() == n);
    assert(N.size2() == m);
    for (i = 0; i < n; i++) {
	for (j = 0; j < m; j++) {
	    assert(N(i, j).first == i);
	    assert(N(i, j).second == j);
	}
    }

    // Confirm assigning works.
    M = matrix<std::pair<size_t, size_t> >(1, 1);
    assert(M.size1() == 1);
    assert(M.size2() == 1);
    M(0, 0) = std::make_pair<size_t, size_t>(123456789, 8);
    assert(M(0, 0).first == 123456789);
    assert(M(0, 0).second == 8);

    // Confirm it did not affect the copy.
    assert(N.size1() == n);
    assert(N.size2() == m);
    for (i = 0; i < n; i++) {
	for (j = 0; j < m; j++) {
	    assert(N(i, j).first == i);
	    assert(N(i, j).second == j);
	}
    }

    // Confirm MatrixD = matrix<double> by confirming the pointer
    // types are compatible.
    matrix<double> MD0(42, 42);
    MatrixD &MD1 = MD0;
    assert(&MD1 == &MD0);

    // Confirm overflow detection.
    try {
	const size_t size_max = std::numeric_limits<size_t>::max();
	MatrixD MD2(size_max, size_max);
    } catch (std::bad_alloc &ba) {
    }
    return 0;
}
