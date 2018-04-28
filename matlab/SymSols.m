function [W]=SymSols(V)
Nx=8; Ny=8; Nz=8; blksize=5;
global Nx Ny Nz blksize
%put the pressure perpendicular to the constant.
cp=ones(Nx*Ny*Nz,1);
for i=1:size(V,2) 
  V(blksize:blksize:end,i)=V(blksize:blksize:end,i)-  ...
         (cp/(cp'*cp))*cp'*V(blksize:blksize:end,i);
end
W=V(:,1:4);
[UW,SW,VW]=svd(W);
diag(SW)
W=UW(:,1:2);
%Symmetry in y-direction W(pos(i,j,k,var),:)*h=W(pos(i,Ny+1-j,k,var),:)h;
k=3;
j=2;
var=1;
for i=1:Nx
	mat(i,:)=  W(pos(i,j,k,var),:)-  W(pos(i,Ny+1-j,k,var),:) ; 
end
null(mat)
[Um,Sm,Vm]=svd(mat);
diag(Sm)
W=W*Vm;
end  
function p=pos(i,j,k,var)
  global Nx Ny Nz blksize
  line=Nx*blksize;
  plane=Ny*line;
  p=var+(i-1)*blksize+(j-1)*line+(k-1)*plane;
end
