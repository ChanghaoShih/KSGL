function D = L_duplication(N)

% a modified duplication matrix that returns negative off-diagonal elements

D = zeros(N*N,N*(N+1)/2);
kb = [1,cumsum((N):-1:1)+1];
for i = 1:N
    D((i-1)*N+i,kb(i)) = 1;
    for j = i+1:N
        D((i-1)*N+j,kb(i)+j-i) = -1;
        D((j-1)*N+i,kb(i)+j-i) = -1;
    end
end


end