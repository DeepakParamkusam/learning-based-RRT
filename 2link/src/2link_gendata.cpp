#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <string>
#include <iostream>

#define pi 3.14159

void gen_data(float bounds[10],int index);
float rand_gen(float min, float max);

void gen_data(float bounds[10], int index)
{
  std::string states_nm, parameters_nm, control_nm;
  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f;

  //  -------------------------------------
  OCP ocp( 0.0, 0.5 );
  // C1
  // ocp.minimizeMayerTerm(T*T);
  // C2
  ocp.minimizeMayerTerm(T*T + tau1*qd1*tau1*qd1 + tau2*qd2*tau2*qd2);

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

  ocp.subjectTo(0.0 <= q1 <= 2*pi);
  ocp.subjectTo(0.0 <= q2 <= 2*pi);
  ocp.subjectTo(-500 <= tau1 <= 500);  // bounds on the control input u,
  ocp.subjectTo(-500 <= tau2 <= 500);

  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm
  algorithm.set(PRINT_COPYRIGHT,BT_FALSE);
  algorithm.set(PRINTLEVEL,LOW);
  algorithm.set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  algorithm.set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);

  algorithm.solve();                        // solves the problem.

  VariablesGrid grid;
  algorithm.getDifferentialStates(grid);
  DVector final_state(4), req_state(4), diff(4);
  final_state = grid.getLastVector();
  req_state(0)=bounds[4];
  req_state(1)=bounds[5];
  req_state(2)=bounds[6];
  req_state(3)=bounds[7];
  diff = final_state - req_state;

  if (diff.getNorm(VN_L2) < 0.001){
    states_nm = "states_"+std::to_string(index)+".txt";
    parameters_nm = "parameters_"+std::to_string(index)+".txt";
    control_nm = "control_"+std::to_string(index)+".txt";
    algorithm.getDifferentialStates(states_nm.c_str());
    algorithm.getObjectiveValue(parameters_nm.c_str());
    algorithm.getControls(control_nm.c_str());
  }
  clearAllStaticCounters();
}

float rand_gen(float lim[2]) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = lim[1] - lim[0];
    float r = random * diff;
    return lim[0] + r;
}

int main(){
  srand (time(NULL));
  int i;
  float q_lims[2] = {0.0,2*pi};
  float qd_lims[2] = {-30.0,30.0};
  float T_lims[2] = {-400,400};

  for(i=0; i<3000; i++){
	  float bounds[10] = {rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims),rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims),T_lims[0],T_lims[1]};
    std::cout << i << std::endl;
    std::cout << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << " " << bounds[6] << " " << bounds[7] << std::endl;
    gen_data(bounds,i);
  }
  return 0;
}
