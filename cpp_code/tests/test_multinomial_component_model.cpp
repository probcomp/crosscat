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
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "MultinomialComponentModel.h"
#include "RandomNumberGenerator.h"
#include "utils.h"


using namespace std;

typedef MultinomialComponentModel MCM;

void insert_elements(MCM& mcm, vector<double> elements) {
    vector<double>::iterator it;
    for (it = elements.begin(); it != elements.end(); it++) {
        mcm.insert_element(*it);
    }
}

void remove_elements(MCM& mcm, vector<double> elements) {
    vector<double>::iterator it;
    for (it = elements.begin(); it != elements.end(); it++) {
        mcm.remove_element(*it);
    }
}

int main() {
    cout << endl << "Begin:: test_multinomial_component_model" << endl;
    RandomNumberGenerator rng;

    // test settings
    int NUM_BUCKETS = 5;
    int num_values_to_test = 30;
    double precision = 1E-10;
    map<string, double> hypers;

    // generate all the random data to use
    //
    // initial parameters
    vector<double> dirichlet_alphas_to_test;
    dirichlet_alphas_to_test.push_back(0.5);
    dirichlet_alphas_to_test.push_back(1.0);
    dirichlet_alphas_to_test.push_back(10.0);
    //
    // elements to add
    vector<double> values_to_test;
    for (int i = 0; i < num_values_to_test; i++) {
        static const double v[] = {
            2,2,3,4,3,4,2,4,2,3,3,1,2,1,4,0,4,1,1,2,3,4,2,2,2,1,4,4,0,1
        };
        values_to_test.push_back(v[i]);
    }
    //
    cout << "values_to_test: " << values_to_test << endl;
    vector<double> values_to_test_reversed = values_to_test;
    std::reverse(values_to_test_reversed.begin(), values_to_test_reversed.end());
    vector<double> values_to_test_shuffled = values_to_test;
    random_shuffle(values_to_test_shuffled.begin(),
                   values_to_test_shuffled.end(),
                   rng);

    // print generated values
    //
    cout << endl << "initial parameters: " << "\t";
    cout << "dirichlet_alphas_to_test: " << dirichlet_alphas_to_test << endl;
    cout << "values_to_test: " << values_to_test << endl;

    hypers["dirichlet_alpha"] = dirichlet_alphas_to_test[1];
    hypers["K"] = NUM_BUCKETS;
    MCM mcm(hypers);

    cout << "calc_marginal_logp() on empty MultinomialComponentModel: ";
    cout << mcm.calc_marginal_logp() << endl;
    assert(is_almost(mcm.calc_marginal_logp(), 0, precision));

    //
    cout << "test insertion and removal in same order" << endl;
    insert_elements(mcm, values_to_test);
    cout << mcm << endl;
    assert(is_almost(mcm.calc_marginal_logp(), -49.9364531937, precision));
    assert(is_almost(mcm.calc_element_predictive_logp(0), log(3.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(1), log(7.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(2), log(10.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(3), log(6.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(4), log(9.0 / 35),
                     precision));
    remove_elements(mcm, values_to_test);
    cout << mcm << endl;
    //
    cout << "test insertion and removal in reversed order" << endl;
    insert_elements(mcm, values_to_test);
    cout << mcm << endl;
    assert(is_almost(mcm.calc_marginal_logp(), -49.9364531937, precision));
    assert(is_almost(mcm.calc_element_predictive_logp(0), log(3.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(1), log(7.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(2), log(10.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(3), log(6.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(4), log(9.0 / 35),
                     precision));
    remove_elements(mcm, values_to_test_reversed);
    cout << mcm << endl;
    //
    cout << "test insertion and removal in shuffled order" << endl;
    insert_elements(mcm, values_to_test);
    cout << mcm << endl;
    assert(is_almost(mcm.calc_marginal_logp(), -49.9364531937, precision));
    assert(is_almost(mcm.calc_element_predictive_logp(0), log(3.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(1), log(7.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(2), log(10.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(3), log(6.0 / 35),
                     precision));
    assert(is_almost(mcm.calc_element_predictive_logp(4), log(9.0 / 35),
                     precision));
    remove_elements(mcm, values_to_test_shuffled);
    cout << mcm << endl;

    cout << "test draws" << endl;
    cout << "inserting: " << values_to_test << endl;
    insert_elements(mcm, values_to_test);
    cout << mcm << endl;
    vector<double> draws;
    map<double, int> draw_counts;
    int num_draws = 10000;
    for (int i = 0; i < num_draws; i++) {
        int rand_int = rng.nexti();
        double draw = mcm.get_draw(rand_int);
        draws.push_back(draw);
        draw_counts[draw]++;
    }
    // cout << "draws are: " << draws << endl;
    cout << "draw_counts is: " << draw_counts << endl;


    cout << endl << endl << "test constructor with sparse input" << endl;
    // elements to add
    values_to_test.clear();
    map<string, double> counts_to_use;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        counts_to_use[stringify(i)] = 0.;
    }
    int ignore_value = 0;
    for (int i = 0; i < num_values_to_test; i++) {
        int rand_i = rng.nexti(NUM_BUCKETS);
        if (rand_i == ignore_value) {
            continue;
        }
        values_to_test.push_back(rand_i);
        counts_to_use[stringify(rand_i)]++;
    }
    counts_to_use.erase(stringify(ignore_value));
    //
    cout << "values_to_test: " << values_to_test << endl;
    cout << "counts_to_use: " << counts_to_use << endl;

    // print generated values
    //
    cout << endl << "initial parameters: " << endl;
    cout << "values_to_test: " << values_to_test << endl;

    hypers["dirichlet_alpha"] = 1.;
    hypers["K"] = NUM_BUCKETS;
    MCM mcm2(hypers, values_to_test.size(), counts_to_use);

    cout << "component model: " << mcm2 << endl;

    draws.clear();
    draw_counts.clear();
    num_draws = 10000;
    for (int i = 0; i < num_draws; i++) {
        int rand_int = rng.nexti();
        double draw = mcm2.get_draw(rand_int);
        draws.push_back(draw);
        draw_counts[draw]++;
    }
    // cout << "draws are: " << draws << endl;
    cout << "draw_counts is: " << draw_counts << endl;

    for (int element = 0; element < NUM_BUCKETS; element++) {
        mcm2.calc_element_predictive_logp(element);
    }

    cout << endl << "End:: test_multinomial_component_model" << endl;
}
