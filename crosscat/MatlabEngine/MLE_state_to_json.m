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
function json_str = MLE_state_to_json(state)
% MLE_state_to_json converts MATLAB code  state to X_L and X_D metadata for use
% with CrossCat

    sample = state;
    column_partitions = order_vector(sample.f);
    row_partitions = gen_row_partitions(column_partitions, sample);

    X_D = gen_X_D(row_partitions);
    X_L = gen_X_L(sample,column_partitions,row_partitions);

    json_str = strcat('{"X_D" : ', X_D, ', "X_L" : ', X_L ,'}');

end

function B = order_vector(A)
% orders the vector from 0 to max(A)-1
    B = zeros(size(A));
    uvals = unique(A);
    
    K = numel(uvals);
    
    for i = 1:K
        a = uvals(i);
        B(A==a) = i-1;
    end
    
    % make sure it worked
    bsort = sort(unique(B));
    assert(~any(bsort-(0:K-1)))
end

function C = vector_to_counts(A)
    U = sort(unique(A));
    K = numel(U);    
    C = hist(A,K);    
end

function row_partitions = gen_row_partitions(column_partitions, sample)
    num_views = max(column_partitions)+1;
    parition_to_order = sample.o;
    row_partitions = [];
    
    unique_views = unique(sample.f);
    for i = 1:numel(sample.f)
        v = sample.f(i);
        index = find(unique_views == v);
        assert(numel(index) <= 1);
        if ~isempty(index)
            row_partitions = [row_partitions; order_vector(parition_to_order(v,:))];
        end
        unique_views(index) = -1;
    end
    assert(all(unique_views==-1))
    assert(size(row_partitions,1) == num_views)
end

function X_D = gen_X_D(row_partitions)
    X_D = array2str(row_partitions);
    if size(row_partitions,1) == 1
        X_D = strcat('[',X_D,']');
    end
end

function X_L = gen_X_L(sample,column_partition,row_partitions)
    
    % column partition hyperparameters
    X_L = '{ "column_partition": {';
    X_L = strcat(X_L,'"hypers" : {"alpha" : ', num2str(sample.crpPriorK), '},');
    X_L = strcat(X_L,'"assignments" : ', array2str(column_partition) , ',');
    X_L = strcat(X_L,'"counts" : ', array2str(vector_to_counts(column_partition)));
    X_L = strcat(X_L,'},');
    
    % column hyperparameters
    hypers = convert_and_get_column_hypers(sample);
    X_L = strcat(X_L,'"column_hypers": ' );
    ch_cell = cell(1,numel(hypers));
    for f = 1:numel(hypers)
        ch_cell{f} = generate_hypers_string(hypers{f});
    end
    
    X_L = strcat(X_L, cell2str(ch_cell),', ');

    % view state
    X_L = strcat(X_L, generate_view_state_string(sample, column_partition,  row_partitions));
    
    X_L = strcat(X_L,'}');

end

function str = generate_row_partition_model_string(which_view, sample, column_partition,  row_partitions)
    v = which_view;
    str = '{"row_partition_model":';
    str = strcat(str, '{"hypers" : ');
    % FIXME: there should be multiple CRP alphas (one for each view), but
    % at this time, there aren't for some reason. 
%     str = strcat(str, '{"alpha":', num2str(sample.crpPriorC(v),'%.10f'),'},');
    if length(sample.crpPriorC) > 1 || length(sample.crpPriorK) > 1
        disp('CRP length > 1')
        keyboard
    end
    str = strcat(str, '{"alpha":', num2str(sample.crpPriorC(1),'%.10f'),'},');
    str = strcat(str, '"counts":', array2str(vector_to_counts(row_partitions(v,:))));
    str = strcat(str, '},');
    
    columns_in_view = sort(find(column_partition==v-1));
    [cm, ~, ~] = gen_cluster_models(v,column_partition,row_partitions,sample);
    
    str = strcat(str, '"column_names": ', array2str(columns_in_view-1) ,', ');
    str = strcat(str, '"column_component_suffstats": ');
    
    % generate cell for column_component_suffstats
    n_cols = numel(columns_in_view);
    n_cats = max(row_partitions(v,:))+1;
    ccs_cell = cell(n_cols, n_cats);
    
    for f = 1:n_cols
        for cat = 1:n_cats
            col = columns_in_view(f);
            assert(cm{col,cat}.N > 0)
            % assert(cm{col,cat}.sum_x_squared > 0)
            % assert(abs(cm{col,cat}.sum_x) > 0)
            ccs_cell{f,cat} = generate_suffstats_string(cm{col,cat});
        end
    end
    
    str = strcat(str,cell2str(ccs_cell));
    str = strcat(str,'}');
    
end

function str = generate_hypers_string(hypers)
    assert( hypers.s > 0);
    assert( hypers.nu > 0);
    assert( hypers.r > 0);
    str = '';
    str = strcat(str,'{ "fixed" : false, ');
    str = strcat(str,'"mu" : ', num2str(hypers.mu,'%.10f') ,', ');
    str = strcat(str,'"nu" : ', num2str(hypers.nu,'%.10f') ,', ');
    str = strcat(str,'"r" : ', num2str(hypers.r,'%.10f') ,', ');
    str = strcat(str,'"s" : ', num2str(hypers.s,'%.10f') ,'}'); 
end

function str = generate_suffstats_string(suffstats)
    assert(suffstats.N > 0)
    % assert(suffstats.sum_x_squared > 0)
    % assert(abs(suffstats.sum_x) > 0)
    str = '';
    str = strcat(str,'{"sum_x" : ', num2str(suffstats.sum_x, '%.10f') ,', ');
    str = strcat(str,'"sum_x_squared" : ', num2str(suffstats.sum_x_squared, '%.10f') ,', ');
    str = strcat(str,'"N" : ', int2str(suffstats.N), '}');
end

function str = generate_view_state_string(sample, column_partition,  row_partitions)
    n_views = max(column_partition)+1;
    vs_cell = cell(1,n_views);
    
    for v = 1:n_views
        vs_cell{v} = generate_row_partition_model_string(v, sample, column_partition,  row_partitions);
    end
    
    str = '"view_state" : ';
    str = strcat(str, cell2str(vs_cell));
end

function hypers = convert_and_get_column_hypers(sample)
% converts the NIG hyperparameters for each feature
    F = sample.F;
    
    hypers = cell(1,F);
    
    % converting from Kevin Murphy's parameterization to Yee Wee Teh's
    for f = 1:F
        hypers{f}.mu = sample.NG_mu(f);
        hypers{f}.s = sample.NG_b(f);
        hypers{f}.nu = 2*sample.NG_a(f);
        hypers{f}.r = sample.NG_k(f);
    end
    
    assert(numel(hypers) == sample.F)
    
end

function [cm, counts, alpha] = gen_cluster_models(which_view,column_partition,row_partitions,sample)
% generates the cluster model for a view.
% -which_view is the view for which we want to generate the model. Goes
% from 1 to n_views.
% -column partitions is a features-length vector where each antry assigns a
% feature (column) to a veiw. Begins at zero.
% -row_partitions is a views by rows array where entry (v,r) is the
% category assignment for the row r of features in view v. Begains at zero.
% -sample is a crosscat state
    R = size(row_partitions,2);
    
    features_in_view = sort(find(column_partition==which_view-1));
    num_features_in_view = numel(features_in_view);
    this_row_partition = row_partitions(which_view,:);
    
    K = max(this_row_partition)+1;
    
    cm = cell(num_features_in_view,K);
    
    counts = zeros(num_features_in_view,K);
    for f = 1:max(features_in_view)
        for k = 1:K
            cm{f,k}.sum_x = 0;
            cm{f,k}.sum_x_squared = 0;
            cm{f,k}.N = 0;
            counts(f,k) = 0;
        end
    end
    
    for f = features_in_view
        for r = 1:R
            k = row_partitions(which_view,r)+1;
            cm{f,k}.sum_x = cm{f,k}.sum_x + sample.data(r,f);
            cm{f,k}.sum_x_squared = cm{f,k}.sum_x_squared + sample.data(r,f)^2;
            cm{f,k}.N = cm{f,k}.N + 1;
        end
    end
    
    % for error checking
    cm2 = cell(num_features_in_view,K);
    for f = features_in_view
        for k = 1:K
            kk = k - 1;
            x = sample.data(row_partitions(which_view,:)==kk,f);
            cm2{f,k}.sum_x = sum(x);
            cm2{f,k}.sum_x_squared = sum(x.^2);
            cm2{f,k}.N = numel(x);
            assert(cm2{f,k}.sum_x == cm{f,k}.sum_x)
            assert(cm2{f,k}.sum_x_squared == cm{f,k}.sum_x_squared)
            assert(cm2{f,k}.N == cm{f,k}.N)
        end
    end
    
%     alpha = sample.crpPriorC(which_view);
    alpha = sample.crpPriorC(1);
    
end

function str = array2str(array)
% turns a matlab array into a JSON-ready string
    [rows,cols] = size(array);
    if rows > 1
        str = '[';
        for r = 1:rows-1
            str = strcat(str,array2str(array(r,:)),', ');
        end
        str = strcat(str,array2str(array(end,:)),']');
    else
        str = '[';
        for c = 1:cols-1
            str = strcat(str,num2str(array(1,c)),', ');
        end
        str = strcat(str,num2str(array(1,end)),']');
    end
end

function str = cell2str(C)
% converts a cell of strings into a JSON-ready string (assuming each string
% is valid JSON)
    [rows,cols] = size(C);
    if rows > 1
        str = '[';
        for r = 1:rows-1
            str = strcat(str,cell2str(C(r,:)),', ');
        end
        str = strcat(str,cell2str(C(end,:)),']');
    else
        str = '[';
        for c = 1:cols-1
            str = strcat(str,C{1,c},', ');
        end
        str = strcat(str,C{1,end},']');
    end

end