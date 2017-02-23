function id=ijk2idx(ii,jj,kk,nx)

  level = floor(log(nx)/log(2));
  if (2^level ~= nx) 
    error('nx must be power of 2');
  end
  
  coord=[ii,jj,kk];

    % octree ordering
    fak2 = 1;
    fak8 = 1;

    id = 0;
    for i = 1:level
      id = id + fak8 * ( 1*mod(floor(coord(1)/fak2),2) ...
                        +2*mod(floor(coord(2)/fak2),2) ...
                        +4*mod(floor(coord(3)/fak2),2) );
      fak2 = fak2 * 2;
      fak8 = fak8 * 8;
    end
end
