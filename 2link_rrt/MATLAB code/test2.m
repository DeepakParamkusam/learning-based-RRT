clear all

load('test4.mat')

iState = data(1,2:5);
dt = 0.025;
final =[iState];

for i=1:20
    j = control_index(i*dt);
    fState = RK4Step(i*dt,iState,dt,control(j,:));
    final = [final;fState];
    iState = fState;
end

plot_test(dt,data,final)
