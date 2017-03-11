clear all; close all;
nx=4;
L=nx-1;

% what to plot
octree_numbering=true;
split_cube=false;
diag_boxes=false;
diag_lines=true;
diag_surf=true;

figure(1),view(3), hold on;
axis([0,L,0,L,0,L]);

if (octree_numbering)

  for i=0:nx-1
    for j=0:nx-1
      for k=0:nx-1
        id=ijk2idx(i,j,k,nx);
        text(i,j,k,num2str(id));
      end
    end
  end

end

if (split_cube)

  X=[0 1 1 0]*L;
  Y=[1 0 0 1]*L;
  Z= [0 0 1 1]*L;


  h=patch('XData',X,'YData',Y,'ZData',Z);
  set(h,'FaceAlpha',0.3);

  h=patch('XData',X,'YData',Z,'ZData',Y);
  set(h,'FaceAlpha',0.3);

  h=patch('XData',Z,'YData',X,'ZData',X);
  set(h,'FaceAlpha',0.3);

end

if (diag_lines)

  plot3([0 L],[0 L],[0 L],'b-');
  plot3([0 L],[L 0],[0 L],'b-');
  plot3([L 0],[0 L],[0 L],'b-');
  
end

if (diag_surf)

  X=[0 0.5 1].*L;
  Y=[0 0.5 0].*L;
  Z=[0 0.5 0].*L;
  C=[1 1 1].*0.7;
  patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C);

  Z=[1 0.5 1].*L;
  patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C);

  X=[0 0.5 0].*L;
  Y=[0 0.5 0].*L;
  Z=[0 0.5 1].*L;
  C=[1 1 1].*0.85;
  patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C);
  
  X=[1 0.5 1].*L;
  patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C);

end

if (diag_boxes) 

  X=[0 1 1 0]-0.5;
  Y=[0 0 1 1]-0.5;
  Z=[0 0 0 0]-0.5;

  for kk=0:nx-1

  XX=X+kk; YY=Y+kk; Z1=Z+kk; Z2=Z1+1;

  h=patch('XData',XX,'YData',YY,'ZData',Z1);
  set(h,'FaceAlpha',0.3);
  h=patch('XData',XX,'YData',YY,'ZData',Z2);
  set(h,'FaceAlpha',0.3);

  h=patch('XData',Z1,'YData',XX,'ZData',YY);
  set(h,'FaceAlpha',0.3);
  h=patch('XData',Z2,'YData',XX,'ZData',YY);
  set(h,'FaceAlpha',0.3);

  h=patch('XData',XX,'YData',Z1,'ZData',YY);
  set(h,'FaceAlpha',0.3);
  h=patch('XData',XX,'YData',Z2,'ZData',YY);
  set(h,'FaceAlpha',0.3);
end

end

axis equal, axis tight, grid on, box on, axis off;
