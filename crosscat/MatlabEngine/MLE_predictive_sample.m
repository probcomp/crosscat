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
function samples = MLE_predictive_sample(X_L, X_D, Y, Q, n)
% MLE_predictive_sample interface wrapper
% Generates predictive samplesin the query Q from a single state defined by X_L
% and X_D
    n = n(1);

    % Just generate samples from hacked code. There is no explicit predictive
    % sample code, so I've taken a chunk that does it from the mutual 
    % information tests
    obj = parse_json(X_L);

    nQ = size(Q,1);

    samples = zeros(nQ,n);

    for i = 1:nQ
        row = Q(i,1);
        col = Q(i,2);
        view = obj.column_partition.assignments{1,col}+1;
        cluster = X_D(view, row)+1;

        index = 0;
        for c = 1:numel(obj.view_state{view}.column_names)
            if obj.view_state{view}.column_names{1,c}+1 == col
                index = c;
                break;
            end
        end

        % get suffstats
        if isstruct(obj.view_state{view}.column_component_suffstats{index})
            X = obj.view_state{view}.column_component_suffstats{index}.sum_x;
            C = obj.view_state{view}.column_component_suffstats{index}.sum_x_squared;
            N = obj.view_state{view}.column_component_suffstats{index}.N;
        else
            X = obj.view_state{view}.column_component_suffstats{index}{cluster}.sum_x;
            C = obj.view_state{view}.column_component_suffstats{index}{cluster}.sum_x_squared;
            N = obj.view_state{view}.column_component_suffstats{index}{cluster}.N;
        end

        % get hypers
        mu = obj.column_hypers{col}.mu;
        s = obj.column_hypers{col}.s;
        r = obj.column_hypers{col}.r;
        nu = obj.column_hypers{col}.nu;

        % build posterior parameters
        r_n = r + N;
        nu_n = nu + N;
        mu_n = (r*mu + X)/r_n;
        s_n = s + C + r*mu^2 - r_n*mu_n^2;

        s_n = s_n / 2 ;
        
        coeff = ((s_n/2)*(r_n+1)) / ((nu_n/2)*r_n);
        
        if coeff <= 0
            disp(coeff)
            disp('mu_n')
            disp(mu_n)
            disp('nu_n')
            disp(nu_n)
            disp('s_n')
            disp(s_n)
            disp('r_n')
            disp(r_n)
            
            assert( coeff > 0 );
        end
        
        
        coeff = sqrt( coeff );
        
        for j = 1:n
            samples(i,j) = trnd(nu_n)*coeff + mu_n;
        end

    end
    if any(~isreal(samples))
        disp('samples')
        disp(samples)
        disp('mu_n')
        disp(mu_n)
        disp('nu_n')
        disp(nu_n)
        disp('s_n')
        disp(s_n)
        disp('r_n')
        disp(r_n)
    end
    samples = samples';

end
