/*
*   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
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
#include "RoundedContinuousComponentModel.h"
#include <boost/math/special_functions/round.hpp>
using namespace std;

double RoundedContinuousComponentModel::get_draw(int random_seed) const {
  vector<double> constraints;
  return get_draw_constrained(random_seed, constraints);
}

double RoundedContinuousComponentModel::get_draw_constrained(int random_seed, vector<double> constraints) const {
  double draw = ContinuousComponentModel::get_draw_constrained(random_seed, constraints);
  draw = boost::math::round(draw);
  return draw;
}
