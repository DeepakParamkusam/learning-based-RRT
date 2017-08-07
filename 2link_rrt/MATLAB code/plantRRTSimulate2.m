function [qdd1,qdd2,fState] = plantRRTSimulate2(time,iState,control)
q1 = iState(1);
q2 = iState(2);
qd1 = iState(3);
qd2 = iState(4);
tau1=control(1);
tau2=control(2);

qdd1 = -(48*tau1 - 48*tau2 + 24*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 18*qd1*qd1*sin(2*q2) - 72*tau2*cos(q2) + 48*qd1*qd2*sin(q2))/(36*cos(q2)*cos(q2) - 64);
qdd2 = (48*tau1 - 240*tau2 + 120*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 36*qd1*qd1*sin(2*q2) + 18*qd2*qd2*sin(2*q2) + 72*tau1*cos(q2) - 144*tau2*cos(q2) + 48*qd1*qd2*sin(q2) + 36*qd1*qd2*sin(2*q2))/(18*cos(2*q2) - 46);

fState = [q1,q2,qd1,qd2];
end