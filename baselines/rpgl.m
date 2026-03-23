function [L,L1,L2] = rpgl(X,param)

N1 = param.N1;
N2 = param.N2;
beta1 = param.beta1;
beta2 = param.beta2;
thr = param.thr;

% calculate mode covariance
X_rs = reshape(X,N2,N1,[]);
S1 = zeros(N1);
S2 = zeros(N2);
for i = 1:size(X,2)
    x = X(:,i);
    Xi = reshape(x, N2, N1);
    S1 = S1 + Xi'*Xi;
    S2 = S2 + Xi*Xi';
end
S1 = S1/size(X,2);
S2 = S2/size(X,2);

% graph learning
D1 = L_duplication(N1);
D2 = L_duplication(N2);

P = blkdiag(2*beta1*(D1'*D1),2*beta2*(D2'*D2));
P_inv = diag(1./diag(P));

q1 = D1'*S1(:);
q2 = D2'*S2(:);
q = [q1;q2];

C1 = [reshape(eye(N1),1,[])*D1;kron(ones(1,N1),eye(N1))*D1];
C2 = [reshape(eye(N2),1,[])*D2;kron(ones(1,N2),eye(N2))*D2];
C = blkdiag(C1,C2);

d1 = [N1;zeros(N1,1)];
d2 = [N2;zeros(N2,1)];
d = [d1;d2];

switch param.solver
    case 'idqp'
        mu = zeros(N1+N2+2,1);
        for i = 1:param.max_iter
            l = P_inv*(C'*mu-q);
            l(l<0) = 0;
            mu_grad = C*l-d;
            mu = mu - param.rho*mu_grad;
            grad_norms(i) = norm(mu_grad,2);
            if norm(mu_grad,2) < param.tol
                break
            end
        end
    case 'qp'
        G = P;
        f = q;
        Aeq = C;
        beq = d;
        lb = zeros(N1*(N1+1)/2+N2*(N2+1)/2,1);
        l = quadprog(G,f,[],[],Aeq,beq,lb,[]);
end
l1 = l(1:N1*(N1+1)/2);
l2 = l(N1*(N1+1)/2+1:end);
L1 = reshape(D1*l1,[N1,N1]);
L2 = reshape(D2*l2,[N2,N2]);

lp1 = l1;
lp2 = l2;
lp1(abs(l1)<thr)=0;
lp2(abs(l2)<thr)=0;
Lp1 = reshape(D1*lp1,[N1,N1]);
Lp2 = reshape(D2*lp2,[N2,N2]);
L = kron(Lp1,eye(N2))+kron(eye(N1),Lp2);

end