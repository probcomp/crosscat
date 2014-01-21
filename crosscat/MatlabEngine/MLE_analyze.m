%
%   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
%
%   Lead Developers: Dan Lovell and Jay Baxter
%   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
%   Research Leads: Vikash Mansinghka, Patrick Shafto
%
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%       http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.
%
function cc_state = MLE_analyze( T, X_L, X_D, specified_s_grid, specified_mu_grid)
% MLE_analyze interface wrapper
% Analyzes a single state defined my X_L and X_D for 1 iteration. Grid 
% arguments are optional

    % init a state, analyze, and return X_L and X_D
    state = MLE_json_to_state(T, X_L, X_D, specified_s_grid, specified_mu_grid);

    state = MLE_do_analyze(state);

    cc_state = MLE_state_to_json(state);
end