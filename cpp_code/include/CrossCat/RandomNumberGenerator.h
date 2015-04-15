/*
 *   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <ctime>
//
#include "constants.h"

/////////////////////////
// from runModel_v2.cpp
class RandomNumberGenerator {
public:
    RandomNumberGenerator() : _engine(0), _dist(_engine) {};
    RandomNumberGenerator(int SEED) : _engine(SEED), _dist(_engine) {};
    double next();
    int nexti(int max = MAX_INT);
    void set_seed(std::time_t seed);
protected:
    // Mersenne Twister
    boost::mt19937  _engine;
    // uniform Distribution
    boost::uniform_01<boost::mt19937> _dist;
};

#endif // GUARD_randomnumbergenerator_h
