% SADE-AMSS
%------------------------------- Reference --------------------------------
% H. Gu, H. Wang, and Y. Jin, Surrogate-Assisted Differential Evolution 
% with Adaptive Multi-Subspace Search for Large-Scale Expensive Optimization 
% in IEEE Transcations on Evolutionary Computation.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 HandingWangXD Group. Permission is granted to copy and
% use this code for research, noncommercial purposes, provided this
% copyright notice is retained and the origin of the code is cited. The
% code is provided "as is" and without any warranties, express or implied.
%---------------------------- Parameter setting ---------------------------
% maxd ---  100  --- Maximum number of variables at each subspace
% Ns   ---  200  --- Initial size of Arc
% tsn  ---  2*d  --- The number of individuals in the training set
% Np   ---   10  --- The size of population
% K    ---   20  --- The maximum number of subspaces in a generation
% Gm   ---    5  --- The maximum iterations of subspace optimization
% tes  ---   50  --- a pre-set cutoff generation
% tr   ---  500  --- a pre-set cutoff generation
% beta ---    2  --- The threshold for switching strategy
% 
% This code is written by Haoran Gu.
% Email: xdu_guhaoran@163.com

clear all
warning off
Fes = 0;                    % function evaluations
Np = 10;                    % the size of population
runnum = 25;                % the number of trial runs
results = zeros(20,runnum); % save result
global lu
global slu
global initial_flag
for func_num = [1,9,20]     % 1:20

        D = 1000;           % dimension
    
                            % Search Range
    if (func_num == 1 | func_num == 4 | func_num == 7 | func_num == 8 | func_num == 9 | func_num == 12 | func_num == 13 | func_num == 14 | func_num == 17 | func_num == 18 | func_num == 19 | func_num == 20)
        xmin = -100;  
        xmax = 100;
    end
    if (func_num == 2 | func_num == 5 | func_num == 10 | func_num == 15)
        xmin = -5;  
        xmax = 5;
    end
    if (func_num == 3 | func_num == 6 | func_num == 11 | func_num == 16)
        xmin = -32;  
        xmax = 32;
    end
    
    MaxFes = 11*D-200;      % maxfes: maximal number of fitness evaluations
    for run = 1:1           % runnum    
        Ns = 200;           % initial size of Arc
        initial_flag = 0;
        rng('shuffle')
        
        K = 20;             % the maximum number of subspaces in a generation
        maxd = 100;         % the maximum number of variables in a subspace
        Gm = 5;             % the maximum iterations of subspace optimization

                            % initialization of DSD
        fsum = 0;
        t = 1;
        Emean = 1;
        DSD = 100000;
        tstop = 0;
        switchflag = 0;
        tes = 50;
        tr = 500; 
        beta = 2;
                            % Begin
                            % Latin hypercube sampling to generate Arc
        Arc = xmin + (xmax - xmin)*lhsdesign(Ns, D);
        Arc(:,D+1) = benchmark_func(Arc(:,1:D),func_num);           

        bestvalue = min(Arc(:,D+1));    

        popt = xmin+2*xmax*lhsdesign(Np,D); % initialization of population

        slu = [xmin*ones(1,D);xmax*ones(1,D)]; % lower and upper boundary points
        
        pop_mean = 0;
        vec_pop = eye(D);

        strflag = 2;
               
        while Fes < MaxFes  
            if mod(Fes+1,strflag) == 0  % mapping subspace
                [M,pop_mean,A_std] = zscore(popt);
                popt = popt-repmat(pop_mean,size(popt,1),1);
                covx = cov(popt);
                [vec_pop,lamd,rate] = pcacov(covx);  % Descending order
                a = lamd(lamd > 1);
                b = lamd(lamd <= 1);
                d_1 = size(a,1);
                popt = popt * vec_pop;  

                slu = [xmin*ones(1,D);xmax*ones(1,D)];
                slu = slu-repmat(pop_mean,size(slu,1),1);   
                slu = slu*vec_pop;  
                slu = sort(slu,1);
            end
            if mod(Fes+1,strflag) ~= 0  % original subspace
                slu = [xmin*ones(1,D);xmax*ones(1,D)];
                pop_mean = 0;
                vec_pop = eye(D);
            end
            k = 1;
            [bestY, index] = min(Arc(:,end));   % bestX for DE
            bestX = Arc(index,1:D);
            bestX = bestX-repmat(pop_mean,size(bestX,1),1); 
            bestX = bestX*vec_pop;          
            while k<=K
                d = randperm(maxd,1);  % the number of decision variables in the k-th subspace
                if mod(Fes+1,strflag) == 0
                    if d > d_1         % Case_1
                        q2 = d-d_1;
                        col_rand2 = randperm(D-d_1);
                        col2 = col_rand2(1:q2);
                        col2 = d_1 + col2(1:q2);
                        col = [1:1:d_1,col2];
                        randIndex = randperm(size(col,2));
                        col = col(:,randIndex);
                    else if d <= d_1   % Case_2
                            col_rand1 = randperm(d);
                            col = col_rand1(1:d);
                        end
                    end
                end
                if mod(Fes+1,strflag) ~= 0
                    col_rand = randperm(D);   
                    col = col_rand(1:d);
                end
                popk = popt(:,col);      
                lu = slu(:,col);        
                tsn = 2*d;     % size of the training set
                [m,n] = size(Arc);

                Xtrain_rand = randperm(m);   
                ip_Xtrain_rand = Xtrain_rand(1:tsn);

                x_kth_trains = Arc(ip_Xtrain_rand,1:D);
                x_kth_trains = x_kth_trains - repmat(pop_mean,size(x_kth_trains,1),1); 
                x_kth_trains = x_kth_trains * vec_pop;       

                x_kth_train = x_kth_trains(:,col);           

                arc_train = Arc(ip_Xtrain_rand,end);
                k_train_point{k} = x_kth_train;

                [lambda{k},gamma{k}] = RBF(x_kth_train,arc_train,'cubic');  % RBFN model
                                                                  
                   
                % Differential evolution
                g = 1; 

                XG = popk;      
                XGf = RBF_eval(XG,x_kth_train,lambda{k},gamma{k}, 'cubic');
                            
                while g <= Gm     
                    XG_next = DE(XG, bestX(1,col));  
                    f_next_G = RBF_eval(XG_next,x_kth_train,lambda{k},gamma{k}, 'cubic');

                    now_popf = [XGf;f_next_G];  
                    [~,y] = sort(now_popf);     
                    now_pop = [XG;XG_next];    
                    % select best Np to the next g
                    XG = now_pop(y(1:Np),:);
                    XGf = now_popf(y(1:Np));

                    g = g + 1;  

                end
                spk = XGf(1);        % best result of k-th subspace
                xbspk = XG(1,:); 
                popt(:,col) = XG;    % update the population
                colfff{k} = col;    
                Dspkfff(k) = d;
                bestspk(k) = spk;   
                bestxbspk{k} =  xbspk;   
                k = k+1;
            end
            xbest_t = bestX;   
            [value_minxbest,pos_minxbest] = min(bestspk);
            kthbestx = bestxbspk{pos_minxbest};         
            bestcol = colfff{pos_minxbest};              
            popt_kth = popt(1,bestcol); 
            out_popt_kth = RBF_eval(popt_kth,k_train_point{pos_minxbest},lambda{pos_minxbest},gamma{pos_minxbest}, 'cubic');
            if out_popt_kth <= value_minxbest   
                xbest_t(:,bestcol) = popt_kth;
            else
                xbest_t(:,bestcol) = kthbestx;
            end
            Fes = Fes + 1;
            % Date space transformation
            xpz = xbest_t*inv(vec_pop); 
            xp = xpz+repmat(pop_mean,size(xpz,1),1);
            for i = 1:D
            if xp(1,i) > xmax
                xp(1,i) = xmax;
            else if xp(1,i) < xmin
                    xp(1,i) = xmin;
                end
            end
            end
            if mod(Fes,strflag) == 0
                popt = popt*inv(vec_pop);  % Date space transformation
                popt = popt+repmat(pop_mean,size(popt,1),1);
                for i = 1:Np
                    for j = 1:D
                        if popt(i,j) > xmax
                            popt(i,j) = xmax;
                        else if popt(i,j) < xmin
                                popt(i,j) = xmin;
                            end
                        end
                    end
                end
            end
            diff = xp; 
            diff(:,D+1) = benchmark_func(diff(:,1:D),func_num);
            Arc = [Arc;diff];     % update the global optimal solution
            bestvalue = min(min(Arc(:,D+1)),bestvalue); 
            fprintf('Fes = %d\tBest fitness: %e\n',Fes,bestvalue);
            % Adaptive switching strategy
            if Fes > 2
                bestvalue1 = log10(min(Arc(1:Fes+Ns-2,end)));
                bestvalue2 = log10(min(Arc(1:Fes+Ns-1,end)));
                bestvalue3 = log10(min(Arc(1:Fes+Ns,end)));
                change1 = bestvalue2 - bestvalue1;
                change2 = bestvalue3 - bestvalue2;
                f_sd(t)=(change2-change1)/2;
                t = t+1;
                if t == 51
                    for i =1:tes
                        fsum = fsum+abs(f_sd(i));
                    end
                    Emean = fsum/tes;
                end
                f_sd_mn  = f_sd/Emean;
                if t > tr+tstop
                    DSD = sum(abs(f_sd_mn(t-tr:t-1)));
                end
                if DSD < beta
                    tstop = Fes+tr; 
                    switchflag = switchflag+1;
                end
                if switchflag == 1
                    strflag = 4;
                    DSD = 100000;
                end
                if switchflag == 2
                    strflag = 1000000000;
                    DSD = 100000;
                end         
            end
        end
        results(func_num, run) = bestvalue;
        dataname =strcat('f_',num2str(func_num,'%d'),'_',num2str(run,'%d'),'.mat');  
        save(dataname,'Arc')
        Fes = 0; 
    end
end
