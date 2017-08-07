iState = [5.68000000000000,1.29000000000000,-20.4200000000000,26.0500000000000];
dt = 0.025;
final =[iState];
val=zeros(6);
for i=1:20
    fState = RK4Step(i*dt,iState,dt,control(i,:));
    final = [final;fState];
    iState = fState;
end

% final
