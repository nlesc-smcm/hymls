clear all; close all;
nx=3; ny=3; nz=3;
%nx=1; ny=1; nz=1;

L=1;
alpha=0.9;
%C1=[1 1 1].*0.6;
%C2=[1 1 1].*0.75;
%C3=[1 1 1].*0.9;

C1=[1 0 0];
C2=[0 1 0];
C3=[0 0 1];


figure(1),view(3), hold on;
axis([0,L,0,L,0,L]);

for i=0:nx-1
  xoff=i*L;
  for j=0:ny-1
    yoff=j*L;
    for k=0:nz-1
      zoff=k*L;
      %plot3([0 L]+xoff,[0 L]+yoff,[0 L]+zoff,'b-');
      %plot3([0 L]+xoff,[L 0]+yoff,[0 L]+zoff,'b-');
      %plot3([L 0]+xoff,[0 L]+yoff,[0 L]+zoff,'b-');
  
      X=[0 0.5 1].*L+xoff;
      Y=[0 0.5 0].*L+yoff;
      Z=[0 0.5 0].*L+zoff;
      C=C1;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

      Z=[1 0.5 1].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

      X=[0 0.5 0].*L+xoff;
      Z=[0 0.5 1].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);
  
      X=[1 0.5 1].*L+xoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      X=[0 0.5 1].*L+xoff;
      Y=[1 0.5 1].*L+yoff;
      Z=[0 0.5 0].*L+zoff;
      C=C2;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

      Z=[1 0.5 1].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

      X=[0 0.5 0].*L+xoff;
      Z=[0 0.5 1].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);
  
      X=[1 0.5 1].*L+xoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      X=[0 0.5 0].*L+xoff;
      Y=[0 0.5 1].*L+yoff;
      Z=[0 0.5 0].*L+zoff;
      C=C3;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

      Z=[1 0.5 1].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

      X=[1 0.5 1].*L+xoff;
      Z=[0 0.5 0].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);
  
      Z=[1 0.5 1].*L+zoff;
      patch('XData',X,'YData',Y,'ZData',Z,'FaceColor',C,'FaceAlpha',alpha);

    end
  end
end

axis equal, axis tight, grid on, box on, axis off;
