clear all;
syms tau1 tau2 q1 q2 qd1 qd2 qdd1 qdd2 m1 m2 r1 r2 l1 l2

Iz1=m1*l1*l1/12;
Iz2=m2*l2*l2/12;

a = Iz1 + Iz2 + m1*r1*r1 + m2*(l1*l1 + r2*r2);
b = m2*l1*r2;
c = Iz2 + m2*r2*r2;

A = [a + 2*b*cos(q2), c + b*cos(q2);
     c + b*cos(q2), c];
B = [-b*sin(q2)*qd2, -b*sin(q2)*(qd1 + qd2);
     b*sin(q2)*qd1, 0];
y=[qd1;qd2];
R=[tau1;tau2];

x=inv(A)*(R-B*y);
x=simplify(expand(x));

qdd1 = eval(subs(x(1),[l1,l2,r1,r2,m1,m2],[1,1,0.5,0.5,1,1]));
qdd2 = eval(subs(x(2),[l1,l2,r1,r2,m1,m2],[1,1,0.5,0.5,1,1]));

file = fopen('qdd.txt', 'w');
fprintf(file, '%s\r\n\n', char(qdd1));
fprintf(file, '%s\r\n\n', char(qdd2));
fclose(file);

