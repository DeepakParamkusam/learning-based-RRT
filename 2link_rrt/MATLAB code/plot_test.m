figure(1);
hold on;
plot(data(:,1), data(:,2), 'r')
plot(data(:,1), data(:,3), 'b')
hold off;
legend('q1', 'q2');

figure(4);
hold on;
plot(data(:,1), rem(final(:,1),2*pi), 'r')
plot(data(:,1), rem(final(:,2),2*pi), 'b')
hold off;
legend('q1', 'q2');