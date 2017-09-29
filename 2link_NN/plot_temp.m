clear all;

knn_error = dlmread('trained_models/knn_error.txt');
% nn_error = dlmread('trained_models/nn_error.txt');
knn_descaled_error = dlmread('trained_models/knn_descaled_error.txt');
% nn_descaled_error = dlmread('trained_models/nn_descaled_error.txt');

knn_error_new = dlmread('trained_models/knn_error_new.txt');
% n_error_new = dlmread('trained_models/nn_error_new.txt');
knn_descaled_error_new = dlmread('trained_models/knn_descaled_error_new.txt');


figure(2)
subplot(2,1,1)
hold on
plot(knn_error)
plot(knn_error_new)
hold off
xlabel('time (\times 0.025)')
ylabel('Scaled torque')
% legend('kNN','NN','True')
legend('knn error','clean knn error')
title('Scaled kNN error')

subplot(2,1,2)
hold on
plot(knn_descaled_error)
plot(knn_descaled_error_new)
hold off
xlabel('time (\times 0.025)')
ylabel('Actual torque')
legend('kNN descaled error','clean descaled knn error')
title('Actual kNN error')

