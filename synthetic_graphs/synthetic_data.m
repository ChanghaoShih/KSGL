clear; close all;
%% set up parameters
addpath(genpath("../utils/sgwt_toolbox/"));
N1 = 20;
N2 = 25;
upper = 2; % range of edge weights
lower = 0.1;
model = 'er'; % 'pa', 'ws', 'grid'
nreplicate = 50;

%% Generate graphs
for ii = 1:nreplicate
    %% generate connected graphs
    while true
        switch model
            case 'er'
                Ap1 = generate_graph(N1,'er',0.3);
            case 'pa'
                Ap1 = generate_graph(N1,'pa',2);
            case 'ws'
                Ap1 = generate_graph(N1,'ws',3,0.1);
            case 'grid'
                Ap1 = generate_graph(N1,'er',4,5);
        end
        if all(conncomp(graph(Ap1))==1)
            break;
        end
    end
    
    while true
        switch model
            case 'er'
                Ap2 = generate_graph(N2,'er',0.3);
            case 'pa'
                Ap2 = generate_graph(N2,'pa',2);
            case 'ws'
                Ap2 = generate_graph(N2,'ws',3,0.1);
            case 'grid'
                Ap2 = generate_graph(N2,'grid',5,5);
        end
        if all(conncomp(graph(Ap2))==1)
            break;
        end
    end
    
    %% Generate the graph Laplacian 
    Lp1_0 = full(sgwt_laplacian(Ap1,'opt','raw'));
    Ap1_0 = -Lp1_0+diag(diag(Lp1_0));
    W = triu(rand(N1)*(upper-lower) + lower, 1);
    Ap1_0 = Ap1_0 .* (W+W');
    Lp1_0 = diag(sum(Ap1_0,1)) - Ap1_0;
    
    Lp2_0 = full(sgwt_laplacian(Ap2,'opt','raw'));
    Ap2_0 = -Lp2_0+diag(diag(Lp2_0));
    W = triu(rand(N2)*(upper-lower) + lower, 1);
    Ap2_0 = Ap2_0 .* (W+W');
    Lp2_0 = diag(sum(Ap2_0,1)) - Ap2_0;
 
    switch pd_type
        case 'cartesian'
            L_0 = kron(Lp1_0,eye(N2)) + kron(eye(N1),Lp2_0);
            A_0 = -L_0+diag(diag(L_0));
        case 'tensor'
            A_0 = kron(Ap1_0,Ap2_0);
            L_0 = diag(sum(A_0))-A_0;
        case 'strong'
            A_0 = kron(Ap1_0,Ap2_0)+kron(Ap1_0,eye(N2))+kron(eye(N1),Ap2_0);
            L_0 = diag(sum(A_0))-A_0;
    end
    
    %% save
    data{ii,1} = A_0;
    data{ii,2} = L_0;
    data{ii,3} = Ap1_0;
    data{ii,4} = Lp1_0;
    data{ii,5} = Ap2_0;
    data{ii,6} = Lp2_0;
end
filename = join([model,"_N1=",num2str(N1),"_N2=",num2str(N2),"_weight=[",num2str(lower),",",num2str(upper),"].mat"],"");
save(filename, 'data')