function plot_test(h,data,final)
t = 0.0:h:0.5;
figure(1);
hold on;
plot(data(:,1), data(:,2), 'r')
plot(t, (final(:,1)), 'g')
hold off
legend('q1-true', 'q1-pred');

figure(2)
hold on
plot(data(:,1), data(:,3), 'b')
plot(t, (final(:,2)), 'k')
hold off;
legend('q2-true', 'q2-pred');

figure(3);
hold on;
plot(data(:,1), data(:,4), 'r')
plot(t, (final(:,3)), 'g')
hold off
legend('qd1-true','qd1-pred');

figure(4)
hold on
plot(data(:,1), data(:,5), 'b')
plot(t, (final(:,4)), 'k')
hold off;
legend('qd2-true', 'qd2-pred');
end