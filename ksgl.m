function [L,L1,L2] = ksgl(X,param)

N1 = param.N1;
N2 = param.N2;
if isfield(param, 'alpha_list')
    alpha1 = param.alpha_list(1);
    alpha2 = param.alpha_list(2);
else
    alpha1 = param.alpha;
    alpha2 = param.alpha;
end
tol = param.tol;
max_iter = param.max_iter;
step_size = param.step_size;
inv_compute = param.inv_compute; 
J = ones(N1*N2)/N1/N2;
p1 = N1*(N1-1)/2;
p2 = N2*(N2-1)/2;

switch param.pd_type
    
    case 'tensor'
        Z = pdist2(X,X,"squaredeuclidean")/size(X,2);
        Z_rs = reshape(Z,N2,N1,N2,N1);
        Z_rs = reshape(permute(Z_rs,[1,3,2,4]),N2*N2,N1*N1);

        S = X*X'/size(X,2);

        w1 = ones(p1,1)/N1;
        w2 = ones(p2,1)/N2;

        W1 = squareform(w1);
        W2 = squareform(w2);
        L1 = diag(sum(W1,1))-W1;
        L2 = diag(sum(W2,1))-W2;
        W = kron(W1,W2);
        L = diag(sum(W,1))-W;

        for k = 1:max_iter

            w10 = w1;
            w20 = w2;
            w0 = w;

            switch param.optim
                case "joint"
                    S1 = -reshape(W2(:)'*Z_rs,N1,N1)/2;
                    S2 = -reshape(Z_rs*W1(:),N2,N2)/2;
                    S1 = S1-diag(diag(S1));
                    S2 = S2-diag(diag(S2));

                    L_gd = -inv(J+L);
                    L_gd_diag = diag(L_gd);
                    L_gd_diag_rs = reshape(L_gd_diag,N2,N1);
                    L_gd_off = L_gd-diag(L_gd_diag);
                    L_gd_off_rs = reshape(L_gd_off,N2,N1,N2,N1);
                    L_gd_off_rs = reshape(permute(L_gd_off_rs,[1,3,2,4]),N2*N2,N1*N1);
                    L1_gd = reshape(W2(:)'*L_gd_off_rs,N1,N1);
                    L1_gd = L1_gd-diag(diag(L1_gd))+diag(diag(L2)'*L_gd_diag_rs);
                    L2_gd = reshape(L_gd_off_rs*W1(:),N2,N2);
                    L2_gd = L2_gd-diag(diag(L2_gd))+diag(L_gd_diag_rs*diag(L1));
        
                    w1_gd = Lstar(S1+L1_gd)+2*alpha1;
                    w2_gd = Lstar(S2+L2_gd)+2*alpha2;
                    w1 = w1-step_size*w1_gd;
                    w2 = w2-step_size*w2_gd;
                    w1(w1<0) = 0;
                    w2(w2<0) = 0;
                    
                    W1 = squareform(w1);
                    W2 = squareform(w2);
                    L1 = diag(sum(W1,1))-W1;
                    L2 = diag(sum(W2,1))-W2;
                    W = kron(W1,W2);
                    L = diag(sum(W,1))-W;
                case "alternate"

                    for i = 1:1
                        S1 = -reshape(W2(:)'*Z_rs,N1,N1)/2;
                        S1 = S1-diag(diag(S1));
    
                        L_gd = -inv(J+L);
                        L_gd_diag = diag(L_gd);
                        L_gd_diag_rs = reshape(L_gd_diag,N2,N1);
                        L_gd_off = L_gd-diag(L_gd_diag);
                        L_gd_off_rs = reshape(L_gd_off,N2,N1,N2,N1);
                        L_gd_off_rs = reshape(permute(L_gd_off_rs,[1,3,2,4]),N2*N2,N1*N1);
    
                        L1_gd = reshape(W2(:)'*L_gd_off_rs,N1,N1);
                        L1_gd = L1_gd-diag(diag(L1_gd))+diag(diag(L2)'*L_gd_diag_rs);
                        w1_gd = Lstar(S1+L1_gd)+2*alpha1;
                        w1 = w1-step_size*w1_gd;
                        w1(w1<0) = 0;
                        W1 = squareform(w1);
                        L1 = diag(sum(W1,1))-W1;
                        W = kron(W1,W2);
                        L = diag(sum(W,1))-W;
                    end

                    if any(isnan(w1))
                        break
                    end

                    for i = 1:1
                        S2 = -reshape(Z_rs*W1(:),N2,N2)/2;
                        S2 = S2-diag(diag(S2));
    
                        L_gd = -inv(J+L);
                        L_gd_diag = diag(L_gd);
                        L_gd_diag_rs = reshape(L_gd_diag,N2,N1);
                        L_gd_off = L_gd-diag(L_gd_diag);
                        L_gd_off_rs = reshape(L_gd_off,N2,N1,N2,N1);
                        L_gd_off_rs = reshape(permute(L_gd_off_rs,[1,3,2,4]),N2*N2,N1*N1);
    
                        L2_gd = reshape(L_gd_off_rs*W1(:),N2,N2);
                        L2_gd = L2_gd-diag(diag(L2_gd))+diag(L_gd_diag_rs*diag(L1));                    
                        w2_gd = Lstar(S2+L2_gd)+2*alpha2;
                        w2 = w2-step_size*w2_gd;
                        w2(w2<0) = 0;
                        W2 = squareform(w2);
                        L2 = diag(sum(W2,1))-W2;
                        W = kron(W1,W2);
                        L = diag(sum(W,1))-W;
                    end

                    if any(isnan(w2))
                        break
                    end

            end
            
            tv(k) = sum(S.*L, 'all');
            w = squareform(W);
            if norm(w-w0,2)/norm(w0,2)<tol
                break
            end
        end

    case 'strong'
        Z = pdist2(X,X,"squaredeuclidean")/size(X,2);
        Z_rs = reshape(Z,N2,N1,N2,N1);
        Z_rs = reshape(permute(Z_rs,[1,3,2,4]),N2*N2,N1*N1);

        S = X*X'/size(X,2);

        w1 = ones(p1,1)/N1;
        w2 = ones(p2,1)/N2;

        W1 = squareform(w1)+eye(N1);
        W2 = squareform(w2)+eye(N2);
        L1 = diag(sum(W1,1))-W1;
        L2 = diag(sum(W2,1))-W2;
        W = kron(W1,W2);
        W = W-diag(diag(W));
        L = diag(sum(W,1))-W;

        for k = 1:max_iter

            w10 = w1;
            w20 = w2;
            w0 = w;

            switch param.optim
                case "alternate"

                    for i = 1:1
                        S1 = -reshape(W2(:)'*Z_rs,N1,N1)/2;
                        S1 = S1-diag(diag(S1));
    
                        L_gd = -inv(J+L);
                        L_gd_diag = diag(L_gd);
                        L_gd_diag_rs = reshape(L_gd_diag,N2,N1);
                        L_gd_off = L_gd-diag(L_gd_diag);
                        L_gd_off_rs = reshape(L_gd_off,N2,N1,N2,N1);
                        L_gd_off_rs = reshape(permute(L_gd_off_rs,[1,3,2,4]),N2*N2,N1*N1);
    
                        L1_gd = reshape(W2(:)'*L_gd_off_rs,N1,N1);
                        L1_gd = L1_gd-diag(diag(L1_gd))+diag((diag(L2)'+1)*L_gd_diag_rs);
                        w1_gd = Lstar(S1+L1_gd)+2*alpha1;
                        w1 = w1-step_size*w1_gd;
                        w1(w1<0) = 0;
                        W1 = squareform(w1)+eye(N1);
                        L1 = diag(sum(W1,1))-W1;
                        W = kron(W1,W2);
                        W = W-diag(diag(W));
                        L = diag(sum(W,1))-W;
                    end

                    if any(isnan(w1))
                        break
                    end

                    for i = 1:1
                        S2 = -reshape(Z_rs*W1(:),N2,N2)/2;
                        S2 = S2-diag(diag(S2));
    
                        L_gd = -inv(J+L);
                        L_gd_diag = diag(L_gd);
                        L_gd_diag_rs = reshape(L_gd_diag,N2,N1);
                        L_gd_off = L_gd-diag(L_gd_diag);
                        L_gd_off_rs = reshape(L_gd_off,N2,N1,N2,N1);
                        L_gd_off_rs = reshape(permute(L_gd_off_rs,[1,3,2,4]),N2*N2,N1*N1);
    
                        L2_gd = reshape(L_gd_off_rs*W1(:),N2,N2);
                        L2_gd = L2_gd-diag(diag(L2_gd))+diag(L_gd_diag_rs*(diag(L1)+1));                    
                        w2_gd = Lstar(S2+L2_gd)+2*alpha2;
                        w2 = w2-step_size*w2_gd;
                        w2(w2<0) = 0;

                        W2 = squareform(w2)+eye(N2);
                        L2 = diag(sum(W2,1))-W2;
                        W = kron(W1,W2);
                        W = W-diag(diag(W));
                        L = diag(sum(W,1))-W;
                    end

                    if any(isnan(w2))
                        break
                    end

            end
            
            tv(k) = sum(S.*L, 'all');
            w = squareform(W);
            if norm(w-w0,2)/norm(w0,2)<tol
                break
            end
        end
end

L = diag(sum(W,1))-W;

end