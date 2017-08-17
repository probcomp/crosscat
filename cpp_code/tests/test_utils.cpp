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
    // >> get_vector_num_blocks([1,2,3,4], {})
    // 4
    map<int, set<int> > block_lookup;
    assert(get_vector_num_blocks(vec, block_lookup) == 4);

    // Each item is in its own block.
    // >> get_vector_num_blocks([1,2,3,4], {1:[1], 2:[2], 3:[3], 4:[4]})
    // 4
    block_lookup.clear();
    int block01[] = {1};
    int block02[] = {2};
    int block03[] = {3};
    int block04[] = {4};
    block_lookup[1] = array_to_set(1, block01);
    block_lookup[2] = array_to_set(1, block02);
    block_lookup[3] = array_to_set(1, block03);
    block_lookup[4] = array_to_set(1, block04);
    assert(get_vector_num_blocks(vec, block_lookup) == 4);

    // One block of two items (1,2), two singletons (3), (4).
    // >> get_vector_num_blocks([1,2,3,4], {1:[1,2], 2:[1,2]})
    // 3
    block_lookup.clear();
    int block11[] = {1,2};
    int block12[] = {1,2};
    block_lookup[1] = array_to_set(2, block11);
    block_lookup[2] = array_to_set(2, block12);
    assert(get_vector_num_blocks(vec, block_lookup) == 3);

    // // Two blocks, each of two items (1,2), (3,4).
    // >> get_vector_num_blocks([1,2,3,4], {1:[1,2], 2:[1,2], 3:[3,4], 4:[3,4]})
    // 2
    block_lookup.clear();
    int block21[] = {1,2};
    int block22[] = {1,2};
    int block23[] = {3,4};
    int block24[] = {3,4};
    block_lookup[1] = array_to_set(2, block21);
    block_lookup[2] = array_to_set(2, block22);
    block_lookup[3] = array_to_set(2, block23);
    block_lookup[4] = array_to_set(2, block24);
    assert(get_vector_num_blocks(vec, block_lookup) == 2);

    // // One block of three items (1,2,4), one singleton block (3).
    // >> get_vector_num_blocks([1,2,3,4], {1:[1,2,3], 2:[1,2,3], 3:[1,2,3]})
    // 2
    block_lookup.clear();
    int block31[] = {1,2,3};
    int block32[] = {1,2,3};
    int block33[] = {1,2,3};
    block_lookup[1] = array_to_set(3, block31);
    block_lookup[2] = array_to_set(3, block32);
    block_lookup[3] = array_to_set(3, block33);
    assert(get_vector_num_blocks(vec, block_lookup) == 2);

    // One block of four items (1,2,3,4)
    // >> get_vector_num_blocks(
    //     [1,2,3,4], {1:[1,2,3,4], 2:[1,2,3,4], 3:[1,2,3,4], 4:[1,2,3,4]})
    // 1
    block_lookup.clear();
    int block41[] = {1,2,3,4};
    int block42[] = {1,2,3,4};
    int block43[] = {1,2,3,4};
    int block44[] = {1,2,3,4};
    block_lookup[1] = array_to_set(4, block41);
    block_lookup[2] = array_to_set(4, block42);
    block_lookup[3] = array_to_set(4, block43);
    block_lookup[4] = array_to_set(4, block44);
    assert(get_vector_num_blocks(vec, block_lookup) == 1);

    // No items.
    vector<int> vecempty;
    assert(get_vector_num_blocks(vecempty, block_lookup) == 0);
}

int main(int argc, char** argv) {
    test_get_vector_num_blocks();
    return 0;
}
