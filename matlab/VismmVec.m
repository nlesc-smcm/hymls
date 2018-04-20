%V=mmread('solution_Rayleigh_Number_1e8.mtx');
%V=mmread('solution_Rayleigh_Number_2015888888.mtx');
%V=mmread('solution_Rayleigh_Number_2002000000.mtx');
nx=8;
V=mmread('solution_Rayleigh_Number_1002000000.mtx');nx=16;
%V=mmread('Roll_x.mm');
%W=SymSol(V);
showV(V,0,nx,nx,nx,5,1)
%mmwrite('roll_x.mm',W(:,1));
%mmwrite('roll_y.mm',W(:,2));

