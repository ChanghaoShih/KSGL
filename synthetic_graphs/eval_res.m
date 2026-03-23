%%% run evaluation on the results from main
addpath("../utils/metrics/")

%% set up the baselines
baselines.pst = 0;
baselines.rpgl = 0;
baselines.ff = 0;
baselines.kglasso = 0;
baselines.teralasso = 0;
baselines.mwgl = 0;
baselines.ksgl = 1;
filter = 'gmrf';
prod = 'tensor'; %'strong' %

mmetric = 1; % the main metric to determine the best parameter

for k = 1:length(Ms)

for ii = 1:nreplicate

L_0 = data{ii,2};
Lp1_0 = data{ii,4};
Lp2_0 = data{ii,6};

Lp1_0 = Lp1_0/trace(Lp1_0)*N1;
Lp2_0 = Lp2_0/trace(Lp2_0)*N2;
Ap1_0 = -Lp1_0+diag(diag(Lp1_0));
Ap2_0 = -Lp2_0+diag(diag(Lp2_0));
L_0 = L_0/trace(L_0)*2*N1*N2;

%% main pst loop
if baselines.pst == 1
    tic;
    
    w1 = res1_pst_graphs1(:,ii,k);
    w2 = res2_pst_graphs2(:,ii,k);
    A1 = squareform(w1);
    A2 = squareform(w2);
    L1 = diag(sum(A1,1))-A1;
    L2 = diag(sum(A2,1))-A2;
    
    A1 = A1/trace(L1)*N1;
    A2 = A2/trace(L2)*N2;
    L1 = L1/trace(L1)*N1;
    L2 = L2/trace(L2)*N2;
    switch prod
        case 'tensor'
            A = kron(A1,A2);
        case 'strong'
            A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
    end
    L = diag(sum(A))-A;

    [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
    res_pst(:,ii) = res;
    res1_pst(:,ii) = res1;
    res2_pst(:,ii) = res2;

    toc;
end

%% main rpgl loop
if baselines.rpgl == 1
    tic;

    len_beta_rpgl1 = size(res1_rpgl_graphs1, 2);
    len_beta_rpgl2 = size(res1_rpgl_graphs1, 3);
    
    for i = 1:len_beta_rpgl1
        for j = 1:len_beta_rpgl2
            w1 = res1_rpgl_graphs1(:,i,j,ii,k);
            w2 = res2_rpgl_graphs2(:,i,j,ii,k);
            A1 = squareform(w1);
            A2 = squareform(w2);
            L1 = diag(sum(A1,1))-A1;
            L2 = diag(sum(A2,1))-A2;
            switch prod
                case 'tensor'
                    A = kron(A1,A2);
                case 'strong'
                    A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
            end
            L = diag(sum(A))-A;

            [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
            res_rpgl(i,j,:,ii) = res;
            res1_rpgl(i,j,:,ii) = res1;
            res2_rpgl(i,j,:,ii) = res2;
        end
    end

    toc;
end

%% main ff loop
if baselines.ff == 1
    tic;
    
    L1 = reshape(res1_ff_graphs1(:,ii,k),N1,N1);
    L2 = reshape(res2_ff_graphs2(:,ii,k),N2,N2);

    L1(L1>0) = 0;
    L2(L2>0) = 0;
    A1 = -L1;
    A2 = -L2;
    L1 = L1-diag(sum(L1,1));
    L2 = L2-diag(sum(L2,1));

    L1 = L1/trace(L1)*N1;
    L2 = L2/trace(L2)*N2;
    switch prod
        case 'tensor'
            A = kron(A1,A2);
        case 'strong'
            A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
    end
    L = diag(sum(A,1))-A;
    L = L/trace(L)*2*N1*N2;

    [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
    res_ff(:,ii) = res;
    res1_ff(:,ii) = res1;
    res2_ff(:,ii) = res2;

    toc;
end

%% main kglasso loop
if baselines.kglasso == 1
    tic;
    
    len_beta_kglasso = size(res1_kglasso_graphs1,2);
    
    for i = 1:len_beta_kglasso
        L1 = reshape(res1_kglasso_graphs1(:,i,ii,k),N1,N1);
        L2 = reshape(res2_kglasso_graphs2(:,i,ii,k),N2,N2);
        
        L1(L1>0) = 0;
        L2(L2>0) = 0;
        A1 = -L1;
        A2 = -L2;
        L1 = L1-diag(sum(L1,1));
        L2 = L2-diag(sum(L2,1));
        L1 = L1/trace(L1)*N1;
        L2 = L2/trace(L2)*N2;
        switch prod
            case 'tensor'
                A = kron(A1,A2);
            case 'strong'
                A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
        end
        L = diag(sum(A,1))-A;
        L = L/trace(L)*2*N1*N2;

        [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
        res_kglasso(i,:,ii) = res; 
        res1_kglasso(i,:,ii) = res1;
        res2_kglasso(i,:,ii) = res2;

    end
    toc;
end

%% main teralasso loop
if baselines.teralasso == 1
    tic;
    
    len_lambda_teralasso1 = length(res1_teralasso_graphs1, 2);
    len_lambda_teralasso2 = length(res1_teralasso_graphs1, 3);
    for i = 1:len_lambda_teralasso1
        for j = 1:len_lambda_teralasso2
    
            L1 = reshape(res1_teralasso_graphs1(:,i,j,ii,k),N1,N1);
            L2 = reshape(res2_teralasso_graphs2(:,i,j,ii,k),N2,N2);
            
            L1(L1>0) = 0;
            L2(L2>0) = 0;
            A1 = -L1;
            A2 = -L2;
            L1 = L1-diag(sum(L1,1));
            L2 = L2-diag(sum(L2,1));
            L1 = L1/trace(L1)*N1;
            L2 = L2/trace(L2)*N2;
            switch prod
                case 'tensor'
                    A = kron(A1,A2);
                case 'strong'
                    A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
            end
            L = diag(sum(A,1))-A;
            L = L/trace(L)*2*N1*N2;

            [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
            res_teralasso(i,j,:,ii) = res;
            res1_teralasso(i,j,:,ii) = res1;
            res2_teralasso(i,j,:,ii) = res2;

        end
        
    end

    toc;
end

%% main mwgl loop
if baselines.mwgl == 1
    tic;
    
    len_alpha_mwgl = length(res1_mwgl_graphs1, 2);
    for i = 1:len_alpha_mwgl
    
        w1 = res1_mwgl_graphs1(:,i,ii,k);
        w2 = res2_mwgl_graphs2(:,i,ii,k);
        A1 = squareform(w1);
        A2 = squareform(w2);
        L1 = diag(sum(A1,1))-A1;
        L2 = diag(sum(A2,1))-A2;

        L1 = L1/trace(L1)*N1;
        L2 = L2/trace(L2)*N2;
        switch prod
            case 'tensor'
                A = kron(A1,A2);
            case 'strong'
                A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
        end
        L = diag(sum(A))-A;
        L = L/trace(L)*2*N1*N2;

        [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
        res_mwgl(i,:,ii) = res; 
        res1_mwgl(i,:,ii) = res1;
        res2_mwgl(i,:,ii) = res2;
        
    end

    toc;
end

%% main ksgl loop
if baselines.ksgl == 1
    tic;
    
    len_alpha_ksgl = size(res1_ksgl_graphs1,2);
    
    for i = 1:len_alpha_ksgl
    
        w1 = res1_ksgl_graphs1(:,1,ii,k);
        w2 = res2_ksgl_graphs2(:,1,ii,k);
        A1 = squareform(w1);
        A2 = squareform(w2);
        L1 = diag(sum(A1,1))-A1;
        L2 = diag(sum(A2,1))-A2;

        L1 = L1/trace(L1)*N1;
        L2 = L2/trace(L2)*N2;
        switch prod
            case 'tensor'
                A = kron(A1,A2);
            case 'strong'
                A = kron(A1,eye(N2))+kron(eye(N1),A2)+kron(A1,A2);
        end
        L = diag(sum(A))-A;
        L = L/trace(L)*2*N1*N2;
        
        [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
        res_ksgl(i,:,ii) = res;
        res1_ksgl(i,:,ii) = res1;
        res2_ksgl(i,:,ii) = res2;
        
    end

    toc;
end

end

%% save

if baselines.pst == 1
    res_pst_best(:,:,k) = evaluate(res_pst,mmetric);
    res1_pst_best(:,:,k) = evaluate(res1_pst,mmetric);
    res2_pst_best(:,:,k) = evaluate(res2_pst,mmetric);
end
if baselines.rpgl == 1
    res_rpgl_best(:,:,k) = evaluate(res_rpgl,mmetric);
    res1_rpgl_best(:,:,k) = evaluate(res1_rpgl,mmetric);
    res2_rpgl_best(:,:,k) = evaluate(res2_rpgl,mmetric);
end
if baselines.ff == 1
    res_ff_best(:,:,k) = evaluate(res_ff,mmetric);
    res1_ff_best(:,:,k) = evaluate(res1_ff,mmetric);
    res2_ff_best(:,:,k) = evaluate(res2_ff,mmetric);
end
if baselines.kglasso == 1
    res_kglasso_best(:,:,k) = evaluate(res_kglasso,mmetric);
    res1_kglasso_best(:,:,k) = evaluate(res1_kglasso,mmetric);
    res2_kglasso_best(:,:,k) = evaluate(res2_kglasso,mmetric);
end
if baselines.teralasso == 1
    res_teralasso_best(:,:,k) = evaluate(res_teralasso,mmetric);
    res1_teralasso_best(:,:,k) = evaluate(res1_teralasso,mmetric);
    res2_teralasso_best(:,:,k) = evaluate(res2_teralasso,mmetric);
end
if baselines.mwgl == 1
    res_mwgl_best(:,:,k) = evaluate(res_mwgl,mmetric);
    res1_mwgl_best(:,:,k) = evaluate(res1_mwgl,mmetric);
    res2_mwgl_best(:,:,k) = evaluate(res2_mwgl,mmetric);
end
if baselines.ksgl == 1
    res_ksgl_best(:,:,k) = evaluate(res_ksgl,mmetric);
    res1_ksgl_best(:,:,k) = evaluate(res1_ksgl,mmetric);
    res2_ksgl_best(:,:,k) = evaluate(res2_ksgl,mmetric);
end

end