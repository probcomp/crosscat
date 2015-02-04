function kl = analyzeData_kl_strengthAndStructure(file, dim1, dim2)
    % assume we are computing P(y|x) vs P(y) where x is dim1, and y is dim2
    % assume fully observed data
    
    % number of iterations
    nSamples = 100;
    
    % load samples
    load(file);

    logPyx = [];
    logPy = [];
    x=[]; y=[];
    xy = []; allC = [];
       
    logPyx = zeros(length(samples), nSamples);
    logPy = zeros(length(samples), nSamples);
    logPx = zeros(length(samples), nSamples);
    
    for ns = 2%1 : length(samples)
        
        if length(unique(samples{ns}.f([dim1 dim2])))==1
        
            for ne = 1 : nSamples

                % draw sample (x,y)
                % choose a sample at random, choose a component, choose (x,y)
                thisSample = ns;

                % sample one category
                cats = unique(samples{thisSample}.o(samples{thisSample}.f(dim1),:));
                prob = [];
                for j = cats
                    prob(end+1) = sum(samples{thisSample}.o(samples{thisSample}.f(dim1),:)==j);
                end
                cats(end+1) = samples{thisSample}.O+1;
                prob(end+1) = samples{thisSample}.crpPriorC;
                prob = normalize(prob);
                prob = cumsum(prob);
                thisCat = cats(find(prob>rand,1));

                % update parameters for t-distribution
                a0 = samples{thisSample}.NG_a([dim1 dim2]);
                b0 = samples{thisSample}.NG_b([dim1 dim2]);
                k0 = samples{thisSample}.NG_k([dim1 dim2]);
                mu0 = samples{thisSample}.NG_mu([dim1 dim2]);

                if thisCat~=(samples{thisSample}.O+1)
                    theseData = samples{thisSample}.o(samples{thisSample}.f(dim1),:)==thisCat;
                    len1 = sum(theseData);
                    len2 = len1;
                    data = samples{thisSample}.data(theseData,:);
                    meanData = mean(data(:,[dim1 dim2]),1);
                    data1 = data(:,dim1);
                    data2 = data(:,dim2);

                    aN = a0 + [len1 len2]./2;
                    kN = k0 + [len1 len2];
                    muN = (k0.*mu0 + [len1 len2].*meanData) ./ (k0+[len1 len2]);
                    bN(1) = b0(1) + .5 .* sum( (data1-repmat(meanData(1),size(data1,1),1)).^2,1) + ...
                                      (k0(1).*len1.*(meanData(1)-mu0(1)).^2 ) ./ ...
                                       (2.*(k0(1)+len1));
                    bN(2) = b0(2) + .5 .* sum( (data2-repmat(meanData(2),size(data2,1),1)).^2,1) + ...
                                      (k0(2).*len2.*(meanData(2)-mu0(2)).^2 ) ./ ...
                                       (2.*(k0(2)+len2));
                else
                    aN = a0;
                    kN = k0;
                    muN = mu0;
                    bN = b0;
                end

                % sample data from a t-distribution
                x(end+1) = trnd(2.*aN(1)) .* (bN(1).*(kN(1)+1))./(aN(1).*kN(1)) + muN(1);
                y(end+1) = trnd(2.*aN(2)) .* (bN(2).*(kN(2)+1))./(aN(2).*kN(2)) + muN(2);
                xy(end+1,:) = [x(end) y(end)];
                allC(end+1) = thisCat;
                
                % choose a sample
                s = ns;            

                % compute p(x), p(y) & p(y,x)

                % loop over categories
                cats = unique(samples{s}.o(samples{s}.f(dim1),:));
                logPc = zeros(1,length(cats)+1);
                logPxGc = zeros(1,length(cats)+1);
                logPyGc = zeros(1,length(cats)+1);
                
                a0 = samples{s}.NG_a([dim1 dim2]);
                b0 = samples{s}.NG_b([dim1 dim2]);
                k0 = samples{s}.NG_k([dim1 dim2]);
                mu0 = samples{s}.NG_mu([dim1 dim2]);
                
                % assess P(x), P(y), and P(x,y)
                for c = 1:length(cats)
                    j = cats(c);

                    theseData = samples{s}.o(samples{s}.f(dim1),:)==j;
                    len = sum(theseData);
                    data = samples{s}.data(theseData,[dim1 dim2]);
                    meanData = mean(data,1);

                    aN = a0 + len./2;
                    kN = k0+len;
                    muN = (k0.*mu0 + len.*meanData) ./ (k0+len);
                    bN = b0 + .5 .* sum( (data-repmat(meanData,size(data,1),1)).^2,1) + ...
                              (k0.*len.*(meanData-mu0).^2 ) ./ ...
                               (2.*(k0+len));

                    % assess probability of category
                    num = sum(samples{s}.o(samples{s}.f(dim1),:)==j);
                    logPc(c) = log(num./(samples{s}.O+samples{s}.crpPriorC));

                    % assess probability of y given c, using t-distribution
                    tmpy = xy(end,2);
                    tmpy = (tmpy-muN(2)) ./ ((bN(2).*(kN(2)+1))./(aN(2).*kN(2)));
                    logPyGc(c) = log(tpdf(tmpy,2.*aN(2)));

                    % assess probability of x given c, using t-distribution
                    tmpx = xy(end,1);
                    tmpx = (tmpx-muN(1)) ./ ((bN(1).*(kN(1)+1))./(aN(1).*kN(1)));
                    logPxGc(c) = log(tpdf(tmpx,2.*aN(1)));
                    
                end
                
                % deal with possiblity of new category!
                logPc(end) = log(samples{s}.crpPriorC./(samples{s}.O+samples{s}.crpPriorC));
                
                tmpy = xy(end,2);
                tmpy = (tmpy-mu0(2)) ./ ( (b0(2).*(k0(2)+1)) ./ (a0(2).*k0(2)) );
                logPyGc(end) = log(tpdf(tmpy,2.*a0(2)));
                
                tmpx = xy(end,1);
                tmpx = (tmpx-mu0(1)) ./ ( (b0(1).*(k0(1)+1)) ./ (a0(1).*k0(1)) );
                logPxGc(end) = log(tpdf(tmpx,2.*a0(1)));
                
                % compute p(y)
                tmpLogPy = logsumexp(logPc+logPyGc);
                
                % compute p(x)
                tmpLogPx = logsumexp(logPc+logPxGc);
                
                % compute p(y,x)
                tmpLogPyx = logsumexp( logPc + logPyGc + logPxGc );

                logPyx(ns,ne) = tmpLogPyx;
                logPy(ns,ne) = tmpLogPy;
                logPx(ns,ne) = tmpLogPx;
                
            end
        end
    end

    kl = logPyx-(logPy+logPx);
    kl = mean(kl,2);
    
    % return linfoot (BAX CHANGE)
    for i = 1:numel(kl)
       mi = kl(i);
       if mi <= 0
           kl(i) = 0;
       else
           kl(i) = sqrt(1-exp(-2*mi));
       end
    end
%     disp(sprintf('Mutual information: %0.3f',kl));
%     disp(sprintf('Standard deviation per sample: %0.3f', mean(std(logPyx-logPy,[],2))));
%     %disp(sprintf('Standard deviation over samples: %0.3f', std(mean(logPyx-logPy,2))));
%     disp(sprintf('Proportion of cases below zero: %0.2f',sum((logPyx(:)-logPy(:))<0)./length(logPy(:))));
end

function [M, z] = normalize(A, dim)
    % NORMALISE Make the entries of a (multidimensional) array sum to 1
    % [M, c] = normalise(A)
    % c is the normalizing constant
    %
    % [M, c] = normalise(A, dim)
    % If dim is specified, we normalise the specified dimension only,
    % otherwise we normalise the whole array.

    if nargin < 2
      z = sum(A(:));
      % Set any zeros to one before dividing
      % This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
      s = z + (z==0);
      M = A / s;
    elseif dim==1 % normalize each column
      z = sum(A);
      s = z + (z==0);
    %   M = A ./ (d'*ones(1,size(A,1)))';
    %   M = A ./ repmatC(s, size(A,1), 1);
      M = A ./ repmat(s, size(A,1), 1);
    %   M = bsxfun(@rdivide,A,s);
    else
      % Keith Battocchi - v. slow because of repmat
      z=sum(A,dim);
      s = z + (z==0);
      L=size(A,dim);
      d=length(size(A));
      v=ones(d,1);
      v(dim)=L;
    %   c=repmat(s,v);
      c=repmat(s,v');
      M=A./c;
    %   M = bsxfun(@rdivide,A,s);
    end
end