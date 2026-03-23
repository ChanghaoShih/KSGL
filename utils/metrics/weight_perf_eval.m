function [re,re_edge_l2,re_edge_l1,re_deg_l2,re_deg_l1] = weight_perf_eval(L_0,L)

N = size(L_0,1);
re = norm(L_0-L,'fro')/norm(L_0,'fro');
edge = L(tril(true(N),-1));
deg = diag(L);
edge_0 = L_0(tril(true(N),-1));
deg_0 = diag(L_0);
re_edge_l1 = norm(edge-edge_0,1)/norm(edge_0,1);
re_edge_l2 = norm(edge-edge_0,2)/norm(edge_0,2);
re_deg_l1 = norm(deg-deg_0,1)/norm(deg_0,1);
re_deg_l2 = norm(deg-deg_0,2)/norm(deg_0,2);

end