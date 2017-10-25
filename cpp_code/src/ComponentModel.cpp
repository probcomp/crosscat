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
#include "ComponentModel.h"

using namespace std;

CM_Hypers ComponentModel::get_hypers() const
{
    return *p_hypers;
}

int ComponentModel::get_count() const
{
    return count;
}

map<string, double> ComponentModel::get_suffstats() const
{
    map<string, double> suffstats_out = _get_suffstats();
    suffstats_out["N"] = count;
    return suffstats_out;
}

std::ostream &operator<<(std::ostream &os, const ComponentModel &cm)
{
    os << cm.to_string() << endl;
    return os;
}

string ComponentModel::to_string(const string &join_str) const
{
    stringstream ss;
    ss << "count: " << count << join_str;
    ss << "suffstats: " << get_suffstats() << join_str;
    ss << "hypers: " << *p_hypers << join_str;
    ss << "marginal logp: " << calc_marginal_logp();
    return ss.str();
}
