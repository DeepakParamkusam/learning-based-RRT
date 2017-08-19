function stateNew = RK4Step(time,state,dt,control)
k1 = eom(time,state,control);
k2 = eom(time + dt/2, state + dt/2*k1,control);
k3 = eom(time + dt/2, state + dt/2*k2,control);
k4 = eom(time + dt, state + dt*k3,control);
stateNew = state + dt/6*(k1+2*k2+2*k3+k4);
end