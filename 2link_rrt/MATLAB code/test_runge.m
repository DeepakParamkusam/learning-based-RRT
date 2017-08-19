clear all;

load('tests/test5.mat');

% h=0.000025;                                             % step size
h = 0.025;
x = 0:h:0.5;                                         % Calculates upto y(3)
y = zeros(length(x),4); 
y(1,:) = data(1,2:5);                    % initial condition

for i=1:(length(x)-1)                              % calculation loop
    j = control_index(x(i));
%     j = i;
    k_1 = eom(x(i),y(i,:),control(j,:));
    k_2 = eom(x(i)+0.5*h,y(i,:)+0.5*h*k_1,control(j,:));
    k_3 = eom((x(i)+0.5*h),(y(i,:)+0.5*h*k_2),control(j,:));
    k_4 = eom((x(i)+h),(y(i,:)+k_3*h),control(j,:));

    y(i+1,:) = y(i,:) + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
end

plot_test(h,data,y)