function runModel(path, dataFile)
    
    nChains = 1;%10;
    nSamples = 5;%5;
    burnIn = 25;%250;
    lag = 10;%50

    % BULID STATE
    % load data
%     state.data = csvread([path, '/Data/',dataFile,'.csv']);
    state.data = csvread(['Data/',dataFile,'.csv']); % BAX CHANGE
    %state.data = mvnrnd([0 0], [1 .9; .9 1], 100);
    %state.data = mvnrnd(zeros(1,100), eye(100), 100);
    
    state.F = size(state.data,2);
    state.O = size(state.data,1);
    
    % parameters
    bins = 31; % must be odd
    x = linspace(.03, .97, bins); %(.03, .97, 30)
    %state.paramPrior = normpdf(x,.5,.5); % normal prior on parameters
    state.paramPrior = ones(1,length(x)); % uniform prior on parameters
    state.paramPrior = state.paramPrior ./ sum(state.paramPrior);
    state.cumParamPrior = cumsum(state.paramPrior);
    state.paramRange = x./(1-x);
    % set CRP parameter ranges
    tmp = linspace(.5, state.F./(state.F+1),(bins+1)/2);
    state.crpKRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    tmp = linspace(.5, state.O./(state.O+1),(bins+1)/2);
    state.crpCRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    state.muRange = linspace(min(state.data(:)),max(state.data(:)),30); % uniform prior
    % set k parameter range
    state.kRange = state.crpCRange;
    % set a range to n/2
    tmp = linspace(.5, (state.O/2)./((state.O/2)+1),(bins+1)/2);
    state.aRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    % set b max based on max empirical SSD
    ssd = max(sum((state.data-repmat(mean(state.data),size(state.data,1),1)).^2));
    tmp = linspace(.5, ssd./(ssd+1),(bins+1)/2);
    state.bRange = [state.paramRange(1:(bins-1)/2), tmp./(1-tmp)];
    % NOTE: all parameter ranges are set based on sufficient stats, such
    % that the max of the range is equal to the max possible from the data
    % we assume a common range for all features
    
    samples = {};
    for nc = 1 : nChains
        disp(nc);
        state.crpPriorK = state.crpKRange(find(state.cumParamPrior>rand,1));
        state.crpPriorC = state.crpCRange(find(state.cumParamPrior>rand,1));
        for i = 1 : state.F
            state.NG_a(i) = state.aRange(find(state.cumParamPrior>rand,1));
            state.NG_k(i) = state.kRange(find(state.cumParamPrior>rand,1));
            state.NG_b(i) = state.bRange(find(state.cumParamPrior>rand,1));
        end
        
        for i = 1 : state.F
            state.NG_mu(i) = state.muRange(randi(length(state.muRange)));
        end

        % initialize state
        state.f = sample_partition(state.F, state.crpPriorK);
        state.o = [];
        for i = 1 : max(state.f)
            state.o(i,:) = sample_partition(state.O, state.crpPriorC);
        end
        %scoreState(state)

        % runModel
        tic
        samples{length(samples)+1} = drawSample(state, burnIn); 
        %scoreState(samples{end})
        toc
        for ns = 2 : nSamples
            samples{end+1} = drawSample(samples{end}, lag);
            %scoreState(samples{end})
            %sum(samples{end}.f==samples{end-1}.f)
            toc
        end
    end
    
    % saveResults
    name = ['crossCatNG_', dataFile,'_',date];
    save([path,'/Samples_cc/',name], 'samples');

end

function state = drawSample(state, lag) 
    
    for i = 1 : lag
        %scoreState(state)
        oldstate = state;
        % sample hyper parameters
        state = sampleHyperParams(state);
        % sample kinds
        if state.F > 1
            state = sampleKinds(state);
        end
        % sample categories
        state = sampleCategories(state);
    end
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~
% SAMPLE HYPER PARAMETERS
%~~~~~~~~~~~~~~~~~~~~~~~~~~
function state = sampleHyperParams(state)
    
    % crpPrior kinds
    logP = zeros(1,length(state.crpKRange));
    for i = 1 : length(state.crpKRange)
        state.crpPriorK = state.crpKRange(i);
        logP(i) = crp(state.f, state.crpPriorK); % only need to look at kinds
    end
    % choose state
    this = chooseState(logP);
    state.crpPriorK = state.crpKRange(this);
    
    % crpPrior categories
    logP = zeros(1,length(state.crpCRange));
    for i = 1 : length(state.crpCRange)
        state.crpPriorC = state.crpCRange(i);
        u = unique(state.f);
        for j = u
            logP(i) = logP(i) + crp(state.o(j,:), state.crpPriorC); % only need to look at categories
        end
    end
    % choose state
    this = chooseState(logP);
    state.crpPriorC = state.crpCRange(this);

    % sample feature params
    for f = 1 : state.F
        [state.NG_a(f) state.NG_k(f) state.NG_b(f)  state.NG_mu(f)] = jumpParam(state, f); 
    end
end

function this = chooseState(logP)
    prob = exp(logP - logsumexp(logP,2));
    cumprob = cumsum(prob);
    this = find(cumprob>rand,1);
end

function [NG_a NG_k NG_b NG_mu] = jumpParam(state, f)

    thisK = state.f(f);
    c = unique(state.o(thisK,:));
    
    NG_a = state.NG_a(f);
    NG_k = state.NG_k(f); 
    NG_b = state.NG_k(f);
    NG_mu = state.NG_mu(f);
    
    NG_mu = sampleMu(state, f);
    NG_a = sampleA(state, f);
    NG_b = sampleB(state, f);
    NG_k = sampleK(state, f);
    
    % k
    function NG_k = sampleK(state, f)
        logP = zeros(1,length(state.kRange));
        for i = 1 : length(state.kRange)
            for j = 1 : length(c)
                theseData = state.o(thisK,:)==c(j);
                logP(i) = logP(i) + NG(state.data(theseData,f), ...
                                       state.NG_mu(f), state.kRange(i), ... 
                                       state.NG_a(f), state.NG_b(f));
                logP(i) = logP(i) + log(state.paramPrior(i));
            end
        end
        % choose state
        this = chooseState(logP);
        NG_k = state.kRange(this)+1;
    end

    % a
    function NG_a = sampleA(state, f)
        logP = zeros(1,length(state.aRange));
        for i = 1 : length(state.aRange)
            for j = 1 : length(c)
                theseData = state.o(thisK,:)==c(j);
                logP(i) = logP(i) + NG(state.data(theseData,f), ...
                                       state.NG_mu(f), state.NG_k(f), ... 
                                       state.aRange(i), state.NG_b(f));
                logP(i) = logP(i) + log(state.paramPrior(i));                               
            end
        end
        % choose state
        this = chooseState(logP);
        NG_a = state.aRange(this);
        %disp(logP); 
        %disp(this);
    end

    % b
    function NG_b = sampleB(state, f)
        logP = zeros(1,length(state.bRange));
        for i = 1 : length(state.bRange)
            for j = 1 : length(c)
                theseData = state.o(thisK,:)==c(j);
                logP(i) = logP(i) + NG(state.data(theseData,f), ...
                                       state.NG_mu(f), state.NG_k(f), ...
                                       state.NG_a(f), state.bRange(i));
                logP(i) = logP(i) + log(state.paramPrior(i));
            end
        end
        % choose state
        this = chooseState(logP);
        NG_b = state.bRange(this);
    end

    % mu
    function NG_mu = sampleMu(state, f)
        logP = zeros(1,length(state.muRange));
        for i = 1 : length(state.muRange)
            for j = 1 : length(c)
                theseData = state.o(thisK,:)==c(j);
                logP(i) = logP(i) + NG(state.data(theseData,f), ...
                                       state.muRange(i), state.NG_k(f), ...
                                       state.NG_a(f), state.NG_b(f));
            end
        end
        % choose state
        this = chooseState(logP);
        NG_mu = state.muRange(this);
    end
    
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~
% SAMPLE KINDS
%~~~~~~~~~~~~~~~~~~~~~~~~~~
function state = sampleKinds(state)  
    
    for f = 1 : state.F
        k = unique(state.f);
        
        % first gibbs (only makes sense if there is more than one feature in this kind, and there is more than one kind)
        if sum(state.f(f)==state.f)>1 && length(k)>1 
            logP = [];
            for K = k
                state.f(f)=K;

                % crp
                sumF = sum(state.f==K);
                if sumF>1
                    logP(end+1) = log( (sumF-1) ./ (state.F-1+state.crpPriorK) );
                else
                    logP(end+1) = log( state.crpPriorK ./ (state.F-1+state.crpPriorK) );
                end

                logP(end) = scoreFeature(state,f); 
            end
            % choose state
            this = chooseState(logP);
            state.f(f) = k(this);
        end
        
        % then MH, choose new v old
        cut = .5; % percent new
        oldState = state;
        newOld = rand>cut;
        
        if length(k)==1 && newOld==1
            continue;
        end
        
        if newOld == 0 % new
%            disp('new');
            logP = [];
            % sample partition
            newK = setdiff(1:state.F+1,k);
            newK = newK(1);
            state.f(f) = newK;
            state.o(newK,:) = sample_partition(state.O, state.crpPriorC);
            
            % score new and score old
            logP(1) = scoreFeature(state, f) + ... % score feature
                      log( state.crpPriorK ./ (state.F-1+state.crpPriorK) ) + ... % new kind
                      crp(state.o(newK,:), state.crpPriorC); % new categories
            logP(2) = scoreFeature(oldState, f) + ... % score feature
                          log( (sum(oldState.f==oldState.f(f))-1) ./ ...
                               (oldState.F-1+oldState.crpPriorK) );
            
            % M-H (t+1 -> t / t -> t+1)
            if sum(oldState.f==oldState.f(f))==1 % deal with single-feature kinds
                % t+1 -> t: prob of new, prob of choosing cat t
                jump(1) = log(cut)+crp(oldState.o(oldState.f(f),:),state.crpPriorC);
                % t -> t+1: prob of new, prob of choosing cat t+1
                jump(2) = log(cut)-crp(state.o(state.f(f),:),state.crpPriorC);
            else
                % t+1 -> t: prob of old, prob of choosing kind @ t+1
                jump(1) = log((1-cut)*(1/length(unique(state.f))));
                % t -> t+1: prob of new, prob of choosing cat t+1
                jump(2) = log(cut)+crp(state.o(newK,:),state.crpPriorC);
            end
            a = logP(1)-logP(2) + jump(1)-jump(2);
            
        else % old
%            disp('old');
            newK = randi(length(k));
            if newK == state.f(f)
                continue;
            end
            logP = [];
            logP(2) = scoreFeature(oldState,f) + ...
                      log( (sum(oldState.f==oldState.f(f))-1) ./ ...
                          (oldState.F-1+oldState.crpPriorK) );
            state.f(f) = newK;
            logP(1) = scoreFeature(state,f) + ...
                      log( sum(state.f==state.f(f))./(state.F-1+state.crpPriorK) );
            
            % M-H tranisition (t+1 -> t / t -> t+1)
            if sum(oldState.f==oldState.f(f))==1 % single feature kind
                % t+1 -> t: prob of new, prob of choosing cat t
                jump(1) = log(cut)+crp(oldState.o(oldState.f(f),:),state.crpPriorC);
                % t -> t+1: prob of old, prob of choosing kind @ t
                jump(2) = log((1-cut)*(1/length(unique(oldState.f))));
                a = logP(1)-logP(2)+jump(1)-jump(2);
            else
                % t+1 -> t: prob of old, prob of choosing kind (same # kinds)
                jump(1) = 0;
                % t -> t+1: prob of old, prob of choosing kind (same # kinds)
                jump(2) = 0;
                a = logP(1)-logP(2) + jump(1)-jump(2);
            end
        end
        
        a = exp(a);
        
%         oldState.f
%         oldState.o
%         state.f
%         state.o
%         
%         fprintf('old logP: %0.2f\n', logP(2));
%         fprintf('new logP: %0.2f\n', logP(1));
%         fprintf('new->old: %0.2f\n', jump(1));
%         fprintf('old->new: %0.2f\n', jump(2));
%        disp(sprintf('%0.2f',a));
        
        if a > rand
            % state is adopted
            %disp('accepted');
        else
            state = oldState;
            %disp('rejected');
        end
                
    end
end

function logP = crp(cats, gama)
    u = unique(cats);
    num = zeros(1,length(u));
    for i = u
        num(i) = sum(cats==i);
    end
    logP = prob_of_partition_via_counts(num, gama);
end

function logP = scoreFeature(state,f)
    % score feature
    K = state.f(f);
    c = unique(state.o(K,:));
    logP = 0;
    for j = c
        theseData = state.o(K,:)==j;
        logP = logP + NG(state.data(theseData,f), ...
                               state.NG_mu(f), state.NG_k(f), ...
                               state.NG_a(f), state.NG_b(f));
    end
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~
% SAMPLE CATEGORIES
%~~~~~~~~~~~~~~~~~~~~~~~~~~
function state = sampleCategories(state)
    k = unique(state.f);
    for K = k
        for O = 1 : state.O
            state = sampleCategoriesK(state,K,O);
        end
    end
end

function state = sampleCategoriesK(state,K,O)
        
    C = unique(state.o(K,:));
    % create a new category
    empty = setdiff(1:state.O, C);
    if isempty(empty) && max(C)==state.O
        % do nothing
    elseif isempty(empty)
        C = [C, length(C)+1];
    else
        C = [C, empty(1)];
    end

    % score alternative categories
    logP = [];
    for c = C
        state.o(K,O) = c;
        logP(end+1) = scoreObject(state,K,O);
    end

    % choose state
    this = chooseState(logP);
    state.o(K,O) = C(this);

end

function logP = scoreObject(state,K,O)
    theseF = find(state.f==K);
    
    % crp
    sumO = sum(state.o(K,:)==state.o(K,O));
    if sumO>1
        logP = log( (sumO-1) ./ (state.O-1+state.crpPriorC) );
    else
        logP = log( state.crpPriorC ./ (state.O-1+state.crpPriorC) );
    end
    %disp(logP);
    
    % score data
    theseData = state.o(K,:)==state.o(K,O);
    theseData(O) = 0; % eliminate this object
    for f = theseF
        logP = logP + NG_cat(state.data(theseData,f), ...
                             state.data(O,f), ...
                             state.NG_mu(f), state.NG_k(f), ...
                             state.NG_a(f), state.NG_b(f) ...
                            );
    end
    
end

function logProb = NG_cat(data, newData, mu0, k0, a0, b0)
    % this is based on kevin murphy's cheat sheet (NG.pdf)
    % data are assumed to be a vector
    % mu0, k0, a0, b0 are hyperparameters
    % NOTE: this version is for the gibbs sampler for categories
    
    % this is updating based on old data
    if isempty(data)
        % do nothing
    else
        % NOTE: this could be cached
        len = length(data);
        meanData = sum(data,1)/len;
        
        mu0 = (k0.*mu0 + len.*meanData) ./ (k0+len);
        k0 = k0+len; 
        a0 = a0 + len./2;
%         b0 = b0 + .5 .* sum( (data-meanData).^2) + ...
%                              (k0.*len.*(meanData-mu0).^2 ) ./ ...
%                               (2.*(k0+len));
        diff1 = data-meanData;
        diff2 = meanData-mu0;
        b0 = b0 + .5 .* sum( diff1.*diff1 ) + ...
                          (k0.*len.*(diff2.*diff2) ) ./ ...
                           (2.*(k0+len));

    end
    
    len = length(newData);
    meanData = sum(newData,1)/len;
    
    % now update with new data
%     muN = (k0.*mu0 + len.*meanData) ./ (k0+len);
    kN = k0+len;
    aN = a0 + len./2;
%     bN = b0 + .5 .* sum( (newData-meanData).^2) + ...
%                          (k0.*len.*(meanData-mu0).^2 ) ./ ...
%                           (2.*(k0+len));
    diff1 = newData-meanData;
    diff2 = meanData-mu0;
    bN = b0 + .5 .* sum( diff1.*diff1 ) + ...
                          (k0.*len.*(diff2.*diff2) ) ./ ...
                           (2.*(k0+len));
    
    logProb = gammaln(aN)-gammaln(a0) + ...
           log(b0).*a0 - log(bN).*aN + ...
           log( (k0./kN) ).*.5  + ...
           log( (2*pi) ).*(-len/2);
       
%     if logProb > 0
%         keyboard
%     end
end

function logP = scoreState(state)
    logP = 0;
    logP = logP + crp(state.f, state.crpPriorK);
    
    F = unique(state.f);
    for f = F
        logP = logP + crp(state.o(f,:),state.crpPriorC);
    end
    
    for f = 1 : state.F
        logP = logP + scoreFeature(state, f);
    end
    
end