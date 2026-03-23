function [w] = Lstar(M)
N = size(M,1);
k = N*(N-1)/2;
w = zeros(k,1);
i = 1;
j = 2;
for l = 1:k
    w(l) = M(i,i) + M(j,j) - M(i,j) - M(j,i);
    if j == N
        i = i+1;
        j = i+1;
    else
        j = j+1;
    end
end
end