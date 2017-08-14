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
#ifndef GUARD_multinomialcomponentmodel_h
#define GUARD_multinomialcomponentmodel_h

#include "ComponentModel.h"
#include "numerics.h"
#include "utils.h"

class MultinomialComponentModel : public ComponentModel
{
public:
    MultinomialComponentModel(const CM_Hypers &in_hypers);
    MultinomialComponentModel(const CM_Hypers &in_hypers,
        int count,
        const std::map<std::string, double> &counts);
    virtual ~MultinomialComponentModel() {};
    //
    // getters
    void get_suffstats(int &count_out,
        std::map<std::string, double> &counts) const;
    std::map<std::string, double> _get_suffstats() const;
    std::map<std::string, double> get_hypers() const;
    void get_keys_counts_for_draw(std::vector<int> &keys,
        std::vector<double> &log_counts_for_draw,
        const std::vector<int> &counts) const;
    double get_draw(int random_seed) const;
    double get_draw_constrained(int random_seed,
        const std::vector<double> &constraints) const;
    //
    // calculators
    double calc_marginal_logp() const;
    double calc_element_predictive_logp(double element) const;
    double calc_element_predictive_logp_constrained(double element,
        const std::vector<double> &constraints) const;
    std::vector<double> calc_hyper_conditionals(const std::string &which_hyper,
        const std::vector<double> &hyper_grid) const;
    //
    // mutators
    double insert_element(double element);
    double remove_element(double element);
    double incorporate_hyper_update();
protected:
    void set_log_Z_0();
    void init_suffstats();
private:
    std::vector<int> suffstats;
    int hyper_K;
    double hyper_dirichlet_alpha;
};

#endif // GUARD_multinomialcomponentmodel_h
