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
function state = MLE_json_to_state(T, X_L, X_D, specified_s_grid, specified_mu_grid)

    % quickly initialize a state, then change the fields
    state = MLE_do_init(T, specified_s_grid, specified_mu_grid);

    % try to get something we can use from JSON
    obj = parse_json(X_L);

    % paritions
    state.f = [obj.column_partition.assignments{1,:}]+1; % columns
    state.o = X_D+1;                              % rows in views

    % extract CRP alphas
    state.crpPriorK = obj.column_partition.hypers.alpha;
    state.crpPriorC = obj.view_state{1,1}.row_partition_model.hypers.alpha;

    % extract column hypers
    for i = 1:state.F
        hypers = obj.column_hypers{1,i};
        
        assert( hypers.s > 0);
        assert( hypers.nu > 0);
        assert( hypers.r > 0);
        
        state.NG_a(i) = hypers.nu/2;
        state.NG_k(i) = hypers.r;
        state.NG_b(i) = hypers.s;
        state.NG_mu(i) = hypers.mu;

    end

end
