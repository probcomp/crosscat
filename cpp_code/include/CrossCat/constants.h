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
#ifndef GUARD_constants_h
#define GUARD_constants_h

#include <limits> // MAX_INT
#include <string>

static const int MAX_INT = std::numeric_limits<int>::max();
static const std::string MULTINOMIAL_DATATYPE = "symmetric_dirichlet_discrete";
static const std::string CONTINUOUS_DATATYPE = "normal_inverse_gamma";
static const std::string CYCLIC_DATATYPE = "vonmises";
static const std::string cyclic_key = "kappa";
static const std::string continuous_key = "nu";
static const std::string multinomial_key = "dirichlet_alpha";
static const std::string TOGETHER = "together";
static const std::string FROM_THE_PRIOR = "from_the_prior";
static const std::string APART = "apart";

#endif // GUARD_constants_h
