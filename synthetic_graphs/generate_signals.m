function [X,X_noisy] = generate_signals(L_0,filter,num_of_signal,eps)

    if nargin < 4
        eps = 0.5;
    elseif nargin < 3
        num_of_signal = 10000;
    elseif nargin < 2
        filter = 'gmrf';
    end

    N = size(L_0,1);
    mu = zeros(1,N);
    [V,D] = eig(full(L_0));
    switch filter
        case 'gmrf'
            sigma = pinv(D);
            gftcoeff = mvnrnd(mu,sigma,num_of_signal);
            X = V*gftcoeff';
        case 'tikhonov'
            sigma = pinv(N+D);
            gftcoeff = mvnrnd(mu,sigma,num_of_signal);
            X = V*gftcoeff';
        case 'diffusion'
            [V,D] = eig(full(L_0));
            sigma = exp(-D);
            gftcoeff = mvnrnd(mu,sigma,num_of_signal);
            X = V*gftcoeff';
    end
    X_noisy = X + eps*randn(size(X));
    
    % source = mvnrnd(mu,eye(N1*N2),num_of_signal);
    % X = (eye(N1*N2)+0.5*A_0)*source';
end