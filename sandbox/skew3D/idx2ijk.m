function [ii,jj,kk]=idx2ijk(id,nx)

  level = floor(log(nx)/log(2));
  if (2^level ~= nx) 
    error('nx must be power of 2');
  end

  fak=[1 2 4];

  bitlevel = 1;
  tElem = mod(id, 8^level);

  coord = zeros(1,3);
  while ( floor(tElem / fak(1)) > 0 )
    coord(:) = coord(:) + bitlevel * floor(mod(tElem./fak(:),2));
    bitlevel = bitlevel * 2;
    fak = fak * 8;
  end
  ii=coord(1); jj=coord(2); kk=coord(3);
end

