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
#ifndef GUARD_randomnumbergenerator_h
#define GUARD_randomnumbergenerator_h

#include <ctime>
#include <stdint.h>

#include "constants.h"
#include "weakprng.h"

class RandomNumberGenerator
{
public:
    RandomNumberGenerator(int seed = 0)
    {
        set_seed(seed);
    }
    double next();
    int nexti(int bound = MAX_INT);
    double stdnormal();
    double stdgamma(double alpha);
    double chisquare(double nu);
    double student_t(double nu);
    void set_seed(std::time_t seed);
protected:
    struct crypto_weakprng _weakprng;
};

#endif // GUARD_randomnumbergenerator_h
