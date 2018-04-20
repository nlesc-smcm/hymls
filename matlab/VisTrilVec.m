V=read_trilinos_vector('EigenBasis.txt');
W=SymSols(V);
showV(W,0,8,8,8,5,1)
%mmwrite('roll_x.mm',W(:,1));
%mmwrite('roll_y.mm',W(:,2));

