[r,c] = size(data);
x0 = 0;
y0 = 0;
figure(4)

for i= 1:r
    q1 = data(i,2);
    q2 = data(i,3);
    x1 = cos(q1);
    y1 = sin(q1);
    x2 = x1 + cos(q1+q2);
    y2 = y1 + sin(q1+q2);
    plot([x0,x1],[y0,y1],'LineWidth',5);
    hold on
    plot([x1,x2],[y1,y2],'LineWidth',5);
    hold off;
    axis([-2 2 -2 2])
    pause(0.1);
end
    