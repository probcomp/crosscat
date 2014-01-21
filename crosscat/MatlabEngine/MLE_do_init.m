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
function state = MLE_do_init(T, specified_s_grid, specified_mu_grid)
    % This is all copied from runModel_independentPriorRanges.m unless noted

    state.data = T; % the only non-copy-pasted code
    
    state.F = size(state.data,2);
    state.O = size(state.data,1);
    
    % parameters
    bins = 31; % must be odd
    x = linspace(.03, .97, bins);
    %Parameter priors are assumed to be uniform in this version
    state.paramPrior = ones(1,length(x)); % uniform prior on parameters
    state.paramPrior = state.paramPrior ./ sum(state.paramPrior);
    state.cumParamPrior = cumsum(state.paramPrior);
    state.paramRange = x./(1-x);
    % set CRP parameter ranges
    tmp = linspace(.5, state.F./(state.F+1),(bins+1)/2);
    state.crpKRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    tmp = linspace(.5, state.O./(state.O+1),(bins+1)/2);
    state.crpCRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    % set k parameter range
    state.kRange = state.crpCRange;
    % set a range to n/2
    tmp = linspace(.5, (state.O/2)./((state.O/2)+1),(bins+1)/2);
    state.aRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    % set mu and beta ranges
    for f = 1 : state.F
        notNan = state.data(~isnan(state.data(:,f)), f);
        if length(notNan)==1 % use all continuous data, here all data
            notNan = state.data(~isnan(state.data));
        end
        % mu
        if length(specified_mu_grid) <= 1
            state.muRange(f,:) = linspace(min(state.data(:,f)),max(state.data(:,f)),30); % uniform prior
        else
            [rows, cols] = size(specified_mu_grid);
            if rows > 1 && cols == 1
                specified_mu_grid = specified_mu_grid';
            end
            state.muRange(f,:) = specified_mu_grid;
        end

        if length(specified_s_grid) <= 1
            % set b max based on max empirical SSD
            ssd = sum((notNan-mean(notNan)).^2);
            % NOTE: assumes ssd is greater than 1!
            tmp = linspace(.5, ssd./(ssd+1),(bins+1)/2);
            % protect against ssd < 1
            if tmp(end) == 0
                tmp(end) = tmp(end-1)/2;
            end
            state.bRange(f,:) = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
            % NOTE: all parameter ranges are set based on sufficient stats, such
            % that the max of the range is equal to the max possible from the data
            % we assume a common range for all features
            
        else
            % make sute that there are no zero entries in s_grid
            [rows, cols] = size(specified_s_grid);
            if rows > 1 && cols == 1
                specified_s_grid = specified_s_grid';
            end
            if any(specified_s_grid==0)
                sgsort = sort(specified_s_grid);
                specified_s_grid(specified_s_grid==0) = sgsort(2)/2;
            end
            state.bRange(f,:) = specified_s_grid;
        end
    end
    
    assert( all(isreal(state.bRange)) )
    assert( all(isreal(state.muRange)) )
    
%     disp('bRange')
%     disp(state.bRange)
%     disp('muRange')
%     disp(state.muRange)

    state.crpPriorK = state.crpKRange(find(state.cumParamPrior>rand,1));
    state.crpPriorC = state.crpCRange(find(state.cumParamPrior>rand,1));

    for i = 1 : state.F
        state.NG_a(i) = state.aRange(find(state.cumParamPrior>rand,1));
        state.NG_k(i) = state.kRange(find(state.cumParamPrior>rand,1));
        state.NG_b(i) = state.bRange(i,find(state.cumParamPrior>rand,1));
    end
    
    for f = 1 : state.F
        state.NG_mu(f) = state.muRange(f,randi(length(state.muRange(f,:))));
    end

    % initialize state
    state.f = sample_partition(state.F, state.crpPriorK);
    state.o = [];
    for i = 1 : max(state.f)
        state.o(i,:) = sample_partition(state.O, state.crpPriorC);
    end

end

function [partition] = sample_partition(n, gama)
% this samples category partions given # objects from crp prior

partition = ones(1,n);
classes = [1,0];

for i=2:n
  classprobs=[];
  
  for j=1:length(classes)
    
    if classes(j) > 0.5
      classprobs(j) = (classes(j))./(i-1+gama);
    else
      classprobs(j) = gama./(i-1+gama);
    end
  
  end
  
  cumclassprobs = cumsum(classprobs);
  c = min(find(rand<cumclassprobs));
  partition(i) = c;
  classes(c)=classes(c)+1;
  
  % if we add new class, need to replace placeholder
  
  if c==length(classes)
    classes(c+1)=0;
  end
  
end
end
