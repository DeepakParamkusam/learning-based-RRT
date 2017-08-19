% clear all;
load('tests/test2.mat')
t=0:0.025:0.5;
figure(1)
plot(t,control(:,1),'r')
figure(2)
plot(t,control(:,2),'r')