clear;close all;

%% load

load("data/Preprocessing_Scripts/data_aligned/y_train_13.mat");
load("data/Preprocessing_Scripts/data_aligned/x_train_13.mat");
load("data/Preprocessing_Scripts/data_aligned/ch_names.mat")

subsample = 1:10:500;
x_train = x_train(:,:,subsample);


N1 = size(x_train,3);
N2 = size(x_train,2);

%% main ksgl loop
for k = 1:4
    tic;
    
    if ~any(y_train==k-1)
        continue
    end
    X = x_train(y_train==k-1,:,:);
    X = reshape(X,size(X,1),N1*N2);
    alpha = [0.1,0.001,0.00001,0];%0.1.^(1:0.2:3);
    len_alpha = length(alpha);
    
    for i = 1:len_alpha
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = alpha(i);
        param.pd_type = 'strong';
        param.inv_compute = 'eig';
        param.optim = "alternate";
        param.max_iter = 10000;
        param.step_size = 1e-2;
        param.tol = 1e-5;
        while true
            [L,L1,L2] = ksgl(X',param);
            if ~any(isnan(L1(:))) && ~any(isnan(L2(:)))
                break
            end
            param.step_size = param.step_size/2;
        end
        A1 = -L1+diag(diag(L1));
        A2 = -L2+diag(diag(L2));
        LL1 = L1;
        LL2 = L2;
        L(abs(L)<10^(-4))=0;
        L1(abs(L1)<0.1)=0;
        L2(abs(L2)<0.1)=0;

        graphs1_ksgl(:,i,k) = -LL1(tril(true(N1),-1));
        graphs2_ksgl(:,i,k) = -LL2(tril(true(N2),-1));
        figure(k)
        subplot(1,2,1);imagesc(A1)
        subplot(1,2,2);imagesc(A2)
        
    end
    toc;
end

%% main mwgl loop
for k = 1:4
    tic;
    if ~any(y_train==k-1)
        continue
    end
    X = x_train(y_train==k-1,:,:);
    X = reshape(X,size(X,1),N1*N2);
    
    alpha = [0.1,0.001,0.00001,0];%
    len_alpha = length(alpha);
    
    for i = 1:len_alpha
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = alpha(i);
        param.pd_type = 'cartesian';
        param.inv_compute = 'eig';
        param.max_iter = 5000;
        param.step_size = 1e-3;
        param.tol = 1e-5;
        [L,L1,L2] = ksgl(X',param);
        LL1 = L1;
        LL2 = L2;
        A1 = -L1+diag(diag(L1));
        A2 = -L2+diag(diag(L2));
        L(abs(L)<10^(-4))=0;
        L1(abs(L1)<0.1)=0;
        L2(abs(L2)<0.1)=0;
        graphs1_mwgl(:,i,k) = -LL1(tril(true(N1),-1));
        graphs2_mwgl(:,i,k) = -LL2(tril(true(N2),-1));
        figure(k)
        subplot(1,2,1);imagesc(A1)
        subplot(1,2,2);imagesc(A2)
    end
    
    toc;
end
