function [L,L1,L2] = mwgl(X,param)

N1 = param.N1;
N2 = param.N2;
alpha1 = param.alpha(1);
alpha2 = param.alpha(2);
tol = param.tol;
max_iter = param.max_iter;
step_size = param.step_size;
inv_compute = param.inv_compute; 
J = ones(N1*N2)/N1/N2;

S = X*X'/size(X,2);
S_rs = reshape(S,N2,N1,N2,N1);
S_rs = permute(S_rs,[1,3,2,4]);

% init
[w1,w2] = init_ws(X,N1,N2);

S2 = zeros(N2);
for j = 1:N1
    S2 = S2+S_rs(:,:,j,j);
end
S_rs = permute(S_rs,[3,4,1,2]);
S1 = zeros(N1);
for i = 1:N2
    S1 = S1+S_rs(:,:,i,i);
end
%         S1 = S1-diag(diag(S1));
%         S2 = S2-diag(diag(S2));
idx1 = sub2ind([N1,N1],1:N1,1:N1);
idx2 = sub2ind([N2,N2],1:N2,1:N2);

W1 = squareform(w1);
W2 = squareform(w2);
L1 = diag(sum(W1,1))-W1;
L2 = diag(sum(W2,1))-W2;
W = kron(W1,eye(N2))+kron(eye(N1),W2);
L = diag(sum(W,1))-W;

for k = 1:max_iter
    
    w10 = w1;
    w20 = w2;

    switch inv_compute
        case 'naive'
            L_gd = -inv(J+L);
            L_gd_rs = reshape(L_gd,N2,N1,N2,N1);
            L_gd_rs = permute(L_gd_rs,[2,4,1,3]);
            L1_gd = sum(L_gd_rs(:,:,idx2),3);
            L_gd_rs = permute(L_gd_rs,[3,4,1,2]);
            L2_gd = sum(L_gd_rs(:,:,idx1),3);
        case 'eig'
            [U1,D1] = eig(L1);
            [U2,D2] = eig(L2);
            D = ones(N2,1)*diag(D1)'+diag(D2);
            D(2:end) = 1./D(2:end);
            L1_gd = -U1*diag(sum(D,1))*U1';
            L2_gd = -U2*diag(sum(D,2))*U2';
    end

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

    tv(k) = sum(S1.*L1, 'all')+sum(S2.*L2,'all');
    c1(k) = norm(w1-w10,2);
    c2(k) = norm(w2-w20,2);
    if norm(w1-w10,2)<tol && norm(w2-w20,2)<tol
        break
    end
end

W = kron(W1,eye(N2))+kron(eye(N1),W2);
L = diag(sum(W,1))-W;


end