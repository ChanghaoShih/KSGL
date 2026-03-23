function [w1, w2] = init_ws(X, N1, N2)
    S1 = zeros(N1);
    S2 = zeros(N2);
    for i = 1:size(X,2)
        x = X(:,i);
        Xi = reshape(x, N2, N1);
        S1 = S1 + Xi'*Xi/N2;
        S2 = S2 + Xi*Xi'/N1;
    end
    S1 = S1/size(X,2);
    S2 = S2/size(X,2);
    Sinv1 = pinv(S1);
    Sinv2 = pinv(S2);
    w1 = -Sinv1(tril(true(N1),-1));
    w1(w1<0) = 0;
    w2 = -Sinv2(tril(true(N2),-1));
    w2(w2<0) = 0;
end