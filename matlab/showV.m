function []=showV(V,type,nz,nx,ny,blksize,CASE);
% nz : number of layers
% nx : number of points in x-direction
% ny ; number of points in y-direction
n=size(V,1);
m=size(V,2);
if 0
switch type
case 'SWE'
  blksize=3; 
case 'MOC'
  blksize=6;
otherwise
 'hope you gave the block size'
end
end
if (nargin < 6)
%automatic detection of sizes
x=fft(V(:,1)); 
[xsort,isort]=sort(abs(x(1:round(n/2))),'descend');
xsort(1:30)
inds=isort(1:30)-1  
rest=round(n./inds)

fac=1
for i=1:3
  %select those numbers that are really possible as product up to a certain accuracy.
  indb=find(abs(n-fac*rest.*inds)<10)
  rest=rest(indb)
  %lowest number has highest accuracy: find the minimum
  ns(i)=min(rest)
  %clean up
  indb=find(rest ~= ns(i))
  rest=rest(indb)
  %divide by the current value
  rest=round(rest/ns(i))
  fac=fac*ns(i) % keeps track of the total factor divided out.
  %update the frequencies based on the current best approximations
  inds=round(n./(rest*fac))
end

blksize=ns(1)
nx=ns(2)
ny=ns(3)
nz=round(n/(nx*ny*blksize))
end
n=nx*nz*ny*blksize; 
mp=min(3,size(V,2));
for i=1:blksize
  figure(i)
  for j=1:mp
    fld=reshape(V(i:blksize:n,j),nx,ny,nz);
    switch CASE 
    case 1
      for k=1:nz    
        subplot(mp,nz,k+(j-1)*nz);
        contourf(fld(:,:,k))
      end
    case 2
      nxl=1
      for k=1:nxl
	subplot(mp,nxl,k+(j-1)*nxl);
	fldtmp(:,:)=fld(k,:,:);
	contourf(fldtmp')
      end
    case 3
      for k=1:ny
	subplot(mp,ny,k+(j-1)*ny);
	fldtmp(:,:)=fld(:,k,:);
	contourf(fldtmp)
      end
    end
  end
end
