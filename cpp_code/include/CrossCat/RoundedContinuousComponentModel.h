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
#ifndef GUARD_roundedcontinuouscomponentmodel_h
#define GUARD_roundedcontinuouscomponentmodel_h

#include "ContinuousComponentModel.h"
#include "numerics.h"
#include "utils.h"


class RoundedContinuousComponentModel : public ContinuousComponentModel {
 public:
  RoundedContinuousComponentModel(CM_Hypers &in_hyper_hash) : ContinuousComponentModel(in_hyper_hash) {};
  RoundedContinuousComponentModel(CM_Hypers &in_hyper_hash,
			   int COUNT, double SUM_X, double SUM_X_SQ) : ContinuousComponentModel(in_hyper_hash, COUNT, SUM_X, SUM_X_SQ) {};
  virtual ~RoundedContinuousComponentModel() {};
  //
  // getters
  double get_draw(int random_seed) const;
  double get_draw_constrained(int random_seed, std::vector<double> constraints) const;
};

#endif // GUARD_roundedcontinuouscomponentmodel_h
