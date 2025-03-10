%% test for TLSp-MSC

clear
close all

addpath(genpath('./ClusteringMeasure'))
addpath(genpath('./Funs'))
data_path = './Data/';


%% Loading data
    fprintf('Testing %s...\n', 'Yale') 
    load(fullfile(data_path, 'yale.mat'));
    views = 3;
    for k=1:views
        eval(sprintf('X{%d} = double(X%d);', k, k));
    end
    cls_num = length(unique(gt));
    K = length(X);

% Run Count
num_runs = 1;

%% Algs Running

        Y = X;
        for iv=1:K
            [Y{iv}]=NormalizeData(X{iv});
        end
 %parameter setting
        opts = [];
        opts.maxIter = 200;
        opts.epsilon = 1e-7;
        opts.flag_debug = 0;
        opts.mu = 1e-5;
        opts.eta = 2;
        opts.max_mu = 1e10;  
        opts.lambda=0.2;

        for kk = 1:num_runs
            time_start = tic;
             [C, S, Out] = alg_TLSp_MSC(Y, cls_num, gt, opts);
            alg_name = 'TLSp-MSC';
            Out.time=toc(time_start);
            alg_cpu(kk) = Out.time;
            alg_NMI(kk) = Out.NMI;
            alg_AR(kk) = Out.AR;
            alg_ACC(kk) = Out.ACC;
            alg_recall(kk) = Out.recall;
            alg_purity(kk) =Out.purity;
            alg_fscore(kk) = Out.fscore;  
            alg_precision(kk) = Out.precision;
        end

 %% Results report
   fprintf('%6s\t%12s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n',...\
           'Stats', 'Algs', 'CPU', 'NMI', 'AR', 'ACC', 'Recall', ...\
           'Pre', 'F-Score', 'Purity');
   fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
         'Mean', alg_name,mean(alg_cpu),mean(alg_NMI),mean(alg_AR),...\
         mean(alg_ACC),mean(alg_recall),mean(alg_precision),...\
         mean(alg_fscore),mean(alg_purity));
  fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
          'Std', alg_name,std(alg_cpu),std(alg_NMI),std(alg_AR),...\
          std(alg_ACC),std(alg_recall),std(alg_precision),...\
          std(alg_fscore),std(alg_purity));


    


