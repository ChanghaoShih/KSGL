function [prauc,precision,recall,Fmeasure,NMI,num_of_edges,deltacon,thr] = structure_perf_eval(L_0,L)
N = size(L_0,1);
w = -L(tril(true(N),-1));
a = L_0(tril(true(N),-1))<0;
[X,Y,T,AUC] = perfcurve(a,w,1, 'XCrit', 'tpr', 'YCrit', 'prec');
fscores = 2*X.*Y./(X+Y);
[Fmeasure,ind] = max(fscores);
precision = Y(ind);
recall = X(ind);
NMI = 0;
num_of_edges = sum(a);
thr = T(ind);
prauc = AUC;
NMI = perfeval_clus_nmi(double(a),double(w>thr));
A0 = -L_0+diag(diag(L_0));
A = -L+diag(diag(L));
deltacon = DeltaConNew('naive',A0>thr,A>thr);
end