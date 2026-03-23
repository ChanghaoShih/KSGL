function [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2)

N1 = size(Lp1_0,1);
N2 = size(Lp2_0,1);

[prauc,precision,recall,Fmeasure,NMI,num_of_edges,thr] = structure_perf_eval(L_0,L);
[prauc1,precision1,recall1,Fmeasure1,NMI1,num_of_edges1,thr1] = structure_perf_eval(Lp1_0,L1);
[prauc2,precision2,recall2,Fmeasure2,NMI2,num_of_edges2,thr2] = structure_perf_eval(Lp2_0,L2);

[re,re_edge_l2,re_edge_l1,re_deg_l2,re_deg_l1] = weight_perf_eval(L_0,L);
[re1,re1_edge_l2,re1_edge_l1,re1_deg_l2,re1_deg_l1] = weight_perf_eval(Lp1_0,L1);
[re2,re2_edge_l2,re2_edge_l1,re2_deg_l2,re2_deg_l1] = weight_perf_eval(Lp2_0,L2);

res = [re,re_edge_l2,re_edge_l1,re_deg_l2,re_deg_l1,prauc,precision,recall,Fmeasure,NMI,num_of_edges];
res1 = [re1,re1_edge_l2,re1_edge_l1,re1_deg_l2,re1_deg_l1,prauc1,precision1,recall1,Fmeasure1,NMI1,num_of_edges1];
res2 = [re2,re2_edge_l2,re2_edge_l1,re2_deg_l2,re2_deg_l1,prauc2,precision2,recall2,Fmeasure2,NMI2,num_of_edges2];

end