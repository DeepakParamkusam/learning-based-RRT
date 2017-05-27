#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <string>
#include <iostream>

#define pi 3.14159

void gen_data(float bounds[14],int index);
float rand_gen(float min, float max);

void gen_data(float bounds[14], int index)
{
  std::string states_nm, parameters_nm, control_nm;
  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f( 0.0, T );

  //  -------------------------------------
  OCP ocp( 0.0, T );
  ocp.minimizeMayerTerm( T );

  f << dot(q1) == qd1;
  f << dot(q2) == qd2;
  f << dot(qd1) == -(20*tau1 - 20*tau2 + 10*qd1*qd1*sin(q2) + 10*qd2*qd2*sin(q2) + 2*qd1*qd1*sin(2*q2) - 8*tau2*cos(q2) + 20*qd1*qd2*sin(q2))/(4*cos(q2)*cos(q2) - 45);
  f << dot(qd2) == (20*tau1 - 56*tau2 + 28*qd1*qd1*sin(q2) + 10*qd2*qd2*sin(q2) + 4*qd1*qd1*sin(2*q2) + 2*qd2*qd2*sin(2*q2) + 8*tau1*cos(q2) - 16*tau2*cos(q2) + 20*qd1*qd2*sin(q2) + 4*qd1*qd2*sin(2*q2))/(2*cos(2*q2) - 43);

  ocp.subjectTo(f);
  ocp.subjectTo(AT_START, q1 == bounds[0] );
  ocp.subjectTo(AT_START, q2 == bounds[1]);
  ocp.subjectTo(AT_START, qd1 == bounds[2]);
  ocp.subjectTo(AT_START, qd2 == bounds[3]);

  ocp.subjectTo(AT_END, q1 == bounds[4]);
  ocp.subjectTo(AT_END, q2 == bounds[5]);
  ocp.subjectTo(AT_END, qd1 == bounds[6]);
  ocp.subjectTo(AT_END, qd2 == bounds[7]);

  ocp.subjectTo(bounds[8] <= tau1 <= bounds[9]);  // bounds on the control input u,
  ocp.subjectTo(bounds[10] <= tau2 <= bounds[11]);
  ocp.subjectTo(bounds[12] <= T <= bounds[13]);     // and the time horizon T.
  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm
  algorithm.set(PRINT_COPYRIGHT,BT_FALSE);
  algorithm.set(PRINTLEVEL,LOW);
  algorithm.set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  algorithm.set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);

  algorithm.solve();                        // solves the problem.

  states_nm = "states_"+std::to_string(index)+".txt";
  parameters_nm = "parameters_"+std::to_string(index)+".txt";
  control_nm = "control_"+std::to_string(index)+".txt";
  algorithm.getDifferentialStates(states_nm.c_str());
  algorithm.getParameters(parameters_nm.c_str());
  algorithm.getControls(control_nm.c_str());

  clearAllStaticCounters();
}

float rand_gen(float lim[2]) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = lim[1] - lim[0];
    float r = random * diff;
    return lim[0] + r;
}

int main(){
  int i;
  float q_lims[2] = {0,pi};
  float T_lims[4] = {-100,100,-100,100};
  float t_horizon[2] = {0.5,10};

  for(i=0; i<2000; i++){
	   float bounds[14] = {rand_gen(q_lims),rand_gen(q_lims),0.0,0.0,rand_gen(q_lims),rand_gen(q_lims),0.0,0.0,T_lims[0],T_lims[1],T_lims[2],T_lims[3],t_horizon[0],t_horizon[1]};
	   gen_data(bounds,i);
  }
  return 0;
}
