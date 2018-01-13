clear all;
syms tau1 tau2 q1 q2 qd1 qd2 t lbd1 lbd2 lbd3 lbd4 tf
assume([tau1,tau2,q1,q2,t,lbd1,lbd2,lbd3,lbd4],'real')

%constants
l1 = 1;
l2 = 1;
r1 = 0.5;
r2 = 0.5;
m1 = 1;
m2 = 1;

%generate eom
Iz1=m1*l1*l1/12;
Iz2=m2*l2*l2/12;

a = Iz1 + Iz2 + m1*r1*r1 + m2*(l1*l1 + r2*r2);
b = m2*l1*r2;
c = Iz2 + m2*r2*r2;

A = [a + 2*b*cos(q2), c + b*cos(q2);
     c + b*cos(q2), c];
B = [-b*sin(q2)*qd2, -b*sin(q2)*(qd1 + qd2);
     b*sin(q2)*qd1, 0];
qd = [qd1;qd2];
u = [tau1;tau2];

qdd = inv(A)*(u-B*qd);
qdd = simplify(expand(qdd));

%state and costate
x1 = q1; x2 = q2; x3 = qd1; x4 = qd2;

x = [x1;x2;x3;x4];
xd = [x3;x4;qdd(1);qdd(2)];
lambda = [lbd1;lbd2;lbd3;lbd4];

%cost and hamiltonian
C = u'*u; %change cost later
H = C + lambda'*xd;

% finding u*
dH_du1 = diff(H,u(1));
dH_du2 = diff(H,u(2));

u1_star = simplify(solve(dH_du1 == 0, u(1)),100);
u2_star = simplify(solve(dH_du2 == 0, u(2)),100);
u_star = [u1_star;u2_star];

%finding diff eq
H_star = subs(H,u,[u1_star;u2_star]);

xd1_star = simplify(diff(H_star,lbd1),100);
xd2_star = simplify(diff(H_star,lbd2),100);
xd3_star = simplify(diff(H_star,lbd3),100);
xd4_star = simplify(diff(H_star,lbd4),100);

lbd1_star = -simplify(diff(H_star,x1),100);
lbd2_star = -simplify(diff(H_star,x2),100);
lbd3_star = -simplify(diff(H_star,x3),100);
lbd4_star = -simplify(diff(H_star,x4),100);

Cd = u_star'*u_star;



