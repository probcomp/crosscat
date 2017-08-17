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

#include <cmath>
#include <limits>
#include <map>
#include <set>

#include "utils.h"

using namespace std;

static void test_get_vector_num_blocks(void) {
    vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    vec.push_back(4);

    // Each item is in its own block.
    map<int, set<int> > block_lookup;
    assert(get_vector_num_blocks(vec, block_lookup) == 4);

    // Somehow figure out how to write these tests concisely.
    // // Each item is in its own block.
    // >> get_vector_num_blocks([1,2,3,4], {})
    // 4
    // >> get_vector_num_blocks([1,2,3,4], {1:[1], 2:[2], 3:[3], 4:[4]})
    // 4
    // // One block of two items (1,2), two singletons (3), (4).
    // >> get_vector_num_blocks([1,2,3,4], {1:[1,2], 2:[1,2]})
    // 3
    // // Two blocks, each of two items (1,2), (3,4).
    // >> get_vector_num_blocks([1,2,3,4], {1:[1,2], 2:[1,2], 3:[3,4], 4:[3,4]})
    // 2
    // // One block of three items (1,2,4), one singleton block (3).
    // >> get_vector_num_blocks([1,2,3,4], {1:[1,2,3], 2:[1,2,3], 3:[1,2,3]})
    // 2
    // // One block of four items (1,2,3,4)
    // >> get_vector_num_blocks(
    //     [1,2,3,4], {1:[1,2,3,4], 2:[1,2,3,4], 3:[1,2,3,4], 4:[1,2,3,4]})
    // 1
}

int main(int argc, char** argv) {
    test_get_vector_num_blocks();
    return 0;
}
