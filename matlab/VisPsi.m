%m-file to create a contourplot after integrating out one direction.
%nx should have been defined by now
x=read_trilinos_vector('x_coords.txt');
y=read_trilinos_vector('y_coords.txt');
z=read_trilinos_vector('z_coords.txt');
rng=1:nx;
dx=x(rng+1)-x(rng);
dy=y(rng+1)-y(rng);
dz=z(rng+1)-z(rng);
e=ones(nx,1);
dx=reshape(kron(kron(e,e),dx),nx,nx,nx);
dy=reshape(kron(kron(e,dy),e),nx,nx,nx);
dz=reshape(kron(kron(dz,e),e),nx,nx,nx);

%Compute fluxes
ufld=reshape(V(1:5:end),nx,nx,nx).*dy.*dz;
vfld=reshape(V(2:5:end),nx,nx,nx).*dx.*dz;
wfld=reshape(V(3:5:end),nx,nx,nx).*dx.*dy;


uAvy=reshape(sum(ufld,2),nx,nx);
%wAvy=reshape(sum(wfld,2),nx,nx);

clear psiuz %this will automatically add zeros 
psiuz(rng+1,rng+1)=cumsum(uAvy,2);

%There is something strange: 
%CONTOUR(Z) draws a contour plot of matrix Z in the x-y plane, with
%    the x-coordinates of the vertices corresponding to column indices
%        of Z and the y-coordinates corresponding to row indices of Z. The
%	    contour levels are chosen automatically.
%Therefore we have to transpose the psi's

figure(1)
contourf(x,z,psiuz');

uAvz=reshape(sum(ufld,3),nx,nx);
%vAvz=reshape(sum(vfld,3),nx,nx);

clear psiuy %this will automatically add zeros 
psiuy(rng+1,rng+1)=cumsum(uAvz,2);

figure(2)
contourf(x,y,psiuy');

%vAvx=reshape(sum(vfld,1),nx,nx);
wAvx=reshape(sum(wfld,1),nx,nx);

clear psiuwy%this will automatically add zeros 
psiwy(rng+1,rng+1)=cumsum(wAvx,1);

figure(3)
contourf(y,z,psiwy');
