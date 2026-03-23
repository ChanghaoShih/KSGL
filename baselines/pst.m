function [A,A1,A2] = pst(X,param)

N1 = param.N1;
N2 = param.N2;
thr = param.thr;

% calculate mode covariance
X_rs = reshape(X,N2,N1,[]);
S1 = mean(pagemtimes(X_rs,'transpose',X_rs,'none'),3)/N2;
S2 = mean(pagemtimes(X_rs,'none',X_rs,'transpose'),3)/N1;

% spectral template based graph learning
switch param.gso
    case 'adjacency'
        A1 = nti(S1,param);
        A2 = nti(S2,param);
    case 'laplacian'
        L1 = nti(S1,param);
        L2 = nti(S2,param);
        A1 = -L1+diag(diag(L1));
        A2 = -L2+diag(diag(L2));
end
Ap1 = A1;
Ap2 = A2;
Ap1(A1<thr) = 0;
Ap2(A2<thr) = 0;

A = kron(Ap1,eye(N2)) + kron(eye(N1),Ap2);

end