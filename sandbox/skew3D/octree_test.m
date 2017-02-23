nx=32;
N=nx^3;

hit_ijk=zeros(nx,nx,nx);
hit_idx=zeros(N,1);

for id=0:N-1
  [i,j,k]=idx2ijk(id,nx);
  hit_ijk(i+1,j+1,k+1)=hit_ijk(i+1,j+1,k+1)+1;
end

t1=sum(sum(sum(hit_ijk)));
t3=0;

for i=0:nx-1
  for j=0:nx-1
    for k=0:nx-1
      id= 1 + ijk2idx(i,j,k,nx);
      hit_idx(id)=hit_idx(id)+1;
      [ii,jj,kk]=idx2ijk(id-1,nx);
      if (ii==i && jj==j && kk==k)
        t3=t3+1;
      end
    end
  end
end

t2=sum(hit_idx);

if (t1==N)
  disp('TEST 1 passed (idx2ijk)');
else
  warning('TEST 1 failed (idx2ijk)');
end

if (t2==N)
  disp('TEST 2 passed (ijk2idx)');
else
  warning('TEST 2 failed (ijk2idx)');
end

if (t3==N)
  disp('TEST 3 passed (reverse ordering)');
else
  warning('TEST 3 failed (reverse ordering)');
end
