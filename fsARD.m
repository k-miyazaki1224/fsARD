function [varargout] = fsARD(G,D,Se,b_init,ag_init, group, wp, ag0, Ntrain,run,trial,maxval)
    %{
    Notation
    N:sensor num
    M:index num
    NG : Number of groups
    
    J:current, [M,T]
    G:leadfield, [N,M]
    D:MEG , [N,T]
    Se:sensor noise covariance,sigma_ep [N,N]
    b_init: initial value of noise variance
    ag_init: initial value of relevance parameter
    group: group information for vertex
    wp : prior weight
    ag0 : prior relevance parameter
    Ntrain: number of training
    run: number of run of data
    trial: number of trial of data
    maxval: threshold of Inf

    b:inverse sensor noise variance, [1,1]
    ag:relevance parameter of all groups [NG,1]
    aj:relevance parameter of all features
    
    
    2021/07/19 O.Yamashita
    2022/10/06 K.Miyazaki add loop for gARD and ARD
    this model use fast-update rule
    This model is uesed for paper %2024/12/26
    %}

    format long

    fprintf('Start groupARD_edKM_ori2_oy_c_fast')
    %G = cat(2,ones(size(G,1),1), G);
    %group para
    %group = [0;group(:)];
    groupid = unique(group); 
    NG = length(groupid);
    %G = cat(2,ones(size(G,1),1),G);

    %ag0 = Inf(NG,1);
    %ag0(2,1) = 1;
    %ag0(182,1) = 1;

    [N,M] = size(G);
    T = size(D,2);

    %
    if nargin < 9
        Ntrain = 2000; 
    end
    if nargin < 8
        ag0 = ones(NG,1); %[1,groupid]
    end
    if nargin < 7
        wp = 0;
    end

    %condition
    if wp == 0;
        maxvalue = maxval; %Inf,1e10
    else
        maxvalue = Inf;
    end
    
    disp(sprintf('wp=%g maxval=%e',wp,maxval))

    %sigma_ep_inv
    iSe = inv(Se); %[N,N]

    %confidence parameter
    gamma_b = 0.5*N*T;
    gamma_a = zeros(NG,1);
    for g = 1:NG
        ix = find(group == groupid(g));
        gamma_a0(g) = (0.5*length(ix)*T*wp) / (1-wp); %hyper para
        gamma_a(g) = 0.5*length(ix) * T + gamma_a0(g);
        index_ig{g} = ix;
    end

    %hyperprior set

    ag_all = zeros(Ntrain,NG);
    b_all = zeros(Ntrain,1);
    C1_all = zeros(Ntrain,1);
    D1_all = zeros(Ntrain,1);

    %%%%% Estimation Loop %%%%%%
    ag = ag_init;
    b = b_init;
    J = ones(M,T);
    k = 1;
    index_a = [];
    loop = false;

    tic
    
    while (k <= Ntrain)

        for l = 1:length(ag)
            if k ~= 1
                if ag_all(k-1,l) < maxvalue && ag(l) ==  0
                    maxval = maxval * 1e-1;
                    loop = true;
                    disp(sprintf('Restart by maxval %e',maxval))
                    [J,ag_all,D1_all,C1_all,b_all,gamma_a,J_all,Sj_all,gamma_all] = fsARD(G,D,Se,b_init,ag_init, group, wp, ag0, Ntrain,run,trial,maxval);
                    break
                end
            end
        end

        if loop == true
            break
        end

        index0g = find(ag >= maxvalue); 
        index1g = setdiff(1:NG, index0g);

        for g = 1:NG
            aj(index_ig{g},1) = ag(g);
        end

        index0 = find(aj >= maxvalue);
        index1 = setdiff(1:M, index0);
        if ~isempty(index0g)
            J(index0,:) = 0;
        end
        %Decision of alpha

        if (length(index1) <= 0)
            disp('Optimization terminated due that all alphas are large.');
            break;
        end

        aj1 = zeros(length(index1),1);
        aj1 = aj(index1,:);
        G1 = zeros(size(G,1),length(index1));
        G1 = G(:,index1);

        ib = 1/b;
        iaj = 1./aj1;
        
         %%%%% J-step %%%%% 
         iAG = (iaj(:,ones(1,N)) .* G1');
         GAG = G1 * iAG; %[N,N]
         GSGC = b*GAG+Se;
         iGSGC = inv(GSGC);
         GSGCD = iGSGC*D;
         J1 = zeros(length(index1),T);
         J1 = b*iAG* GSGCD;
         J(index1,:) = J1;
         Sj1 = zeros(length(index1));
         Sj1 = iaj - iaj.^2 .* sum((b*G1'*iGSGC).*G1',2); %fast3
         Sj(index1,:) = Sj1;
        
        %%%%% B-step %%%%%
        gamma1 = zeros(length(index1),1);
        gamma1 = 1 - aj1.*Sj1;
        gamma(index1,:) = gamma1;
        C1 = sum(sum(Se' .* (GSGCD*GSGCD')'));
        D1 = (N - sum(gamma1))*T;

        ib = C1/D1;
        b = 1/ib;

        %%%%% A-step %%%%%
        dSj = Sj;
        
        for g = 1:length(index1g) %NG
            C2(g) = sum( gamma(index_ig{index1g(g)}) + wp.*aj(index_ig{index1g(g)},1).*dSj(index_ig{index1g(g)}))*T;
            D2(g) = (wp/ag0(index1g(g)))*T*length(index_ig{index1g(g)}) + (1-wp).*sum(sum(J(index_ig{index1g(g)},:).^2));
            ag(index1g(g)) = C2(g)/D2(g);
        end

        %for save
        J_all(:,:,k) = J;
        Sj_all(k,:) = dSj;
        ag_all(k,:) = ag;
        b_all(k,1) = b;
        C1_all(k,1) = C1;
        D1_all(k,1) = D1;
        gamma_all(k,:) = gamma;
        C2_all(k,:) = C2;
        D2_all(k,:) = D2;
        

        if mod(k, 200)==0 || k==Ntrain
            %toc 
            fprintf('iter%03d min(a)=10^%0.2f max(a)=10^%0.2f b=%0.2f index1g=%03d index1=%03d \n', k, min(log10(aj)), max(log10(aj)), b, length(index1g),length(index1));
            %tic
        end

        %save
        index_a = find(ag == 0);
        k = k+1;
    end

    if loop == true
        varargout{1} = J;
        varargout{2} = ag_all;
        varargout{3} = D1_all;
        varargout{4} = C1_all;
        varargout{5} = b_all;
        varargout{6} = gamma_a;
        varargout{7} = J_all;
        varargout{8} = Sj_all;
        varargout{9} = gamma_all;
    else
        if(k < Ntrain)
            if isempty(index_a)
                fprintf('Optimization of alpha and beta successfull.\n');
            else
                fprintf('exist index_a \n');
            end
        else
            if isempty(index_a)
                fprintf('Optimization terminated due to max iteration.\n');
            else
                fprintf('exist index_a \n');
            end
        end

        %out
        varargout{1} = J;
        varargout{2} = ag_all;
        varargout{3} = D1_all;
        varargout{4} = C1_all;
        varargout{5} = b_all;
        varargout{6} = gamma_a;
        varargout{7} = J_all;
        varargout{8} = Sj_all;
        varargout{9} = gamma_all;
    end
end