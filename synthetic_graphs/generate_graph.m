function G = generate_graph(N,opt,varargin1,varargin2)
% Graph construction

%% check inputs
if nargin == 2
    if strcmp(opt,'chain') == 0
        error('number of input variables not correct :(')
    end
elseif nargin == 3
    if strcmp(opt,'ff') || strcmp(opt,'grid')
        error('number of input variables not correct :(')
    end
elseif nargin == 4
    if strcmp(opt,'er') || strcmp(opt,'pa')
        error('number of input variables not correct :(')
    end
end

%% construct the graph
switch opt
        
    case 'er', % Erdos-Renyi random graph
        p = varargin1;
        G = erdos_reyni(N,p);
        
    case 'pa', % scale-free graph with preferential attachment
        m = varargin1;
        G = preferential_attachment_graph(N,m);
        
    case 'ff', % forest-fire model
        p = varargin1;
        r = varargin2;
        G = forest_fire_graph(N,p,r);
        
    case 'chain' % chain graph
        G = spdiags(ones(N-1,1),-1,N,N);
        G = G + G';

    case 'grid' % grid graph
        a = varargin1;
        b = varargin2;
        G1 = spdiags(ones(a-1,1),-1,a,a);
        G1 = G1 + G1';
        G2 = spdiags(ones(b-1,1),-1,b,b);
        G2 = G2 + G2';
        G = kron(G1,eye(b))+kron(eye(a),G2);

    case 'sbm'
        p = varargin1;
        q = varargin2;
        G1 = erdos_reyni(N/2,p);
        G2 = erdos_reyni(N/2,p);
        G3 = rand(N/2)<q;
        G = [G1,G3;G3',G2];

    case 'ws' % small world network
        % Connect each node to its K next and previous neighbors. This constructs
        % indices for a ring lattice.
        K = varargin1;
        p = varargin2;
        s = repelem((1:N)',1,K);
        t = s + repmat(1:K,N,1);
        t = mod(t-1,N)+1;

        % Rewire the target node of each edge with probability p
        for source=1:N    
            switchEdge = rand(K, 1) < p;
            
            newTargets = rand(N, 1);
            newTargets(source) = 0;
            newTargets(s(t==source)) = 0;
            newTargets(t(source, ~switchEdge)) = 0;
            
            [~, ind] = sort(newTargets, 'descend');
            t(source, switchEdge) = ind(1:nnz(switchEdge));
        end

        G = zeros(N);
        idx = sub2ind(size(G), s(:),t(:));
        G(idx) = 1;
        G = G + G';
end