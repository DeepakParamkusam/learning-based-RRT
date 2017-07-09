figure(1);
hold on;
plot(data(:,1), data(:,2), 'r')
plot(data(:,1), data(:,3), 'b')
hold off;
legend('q1', 'q2');

figure(2);
hold on;
plot(data(:,1), data(:,4), 'r')
plot(data(:,1), data(:,5), 'b')
hold off;
legend('qd1', 'qd2');

figure(3);
hold on;
plot(data(:,1), data(:,6), 'r')
plot(data(:,1), data(:,7), 'b')
hold off;
legend('tau1', 'tau2');