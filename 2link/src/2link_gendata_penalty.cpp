#include <iostream>
#include <string>
#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <cmath>

#define pi 3.14159
#define E 2.718

using namespace std;

int gen_data(float bounds[8],int index, char cost_func);
float rand_gen(float min, float max);

int gen_data(float bounds[8], int index, char cost_func)
{
  string states_nm, parameters_nm, control_nm;
  bool flag = false;
  double inf_val = 1.0e20;

  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f;

  //  -------------------------------------
  OCP ocp( 0.0, 0.5 );
  switch(cost_func){
    case 'a': ocp.minimizeMayerTerm(T*T + (1/(600*600-tau1*tau1)) + (1/(600*600-tau2*tau2)));
              cout << "cost function = sq_power + inverse penalty " << endl;
              break;

    default: ocp.minimizeMayerTerm(T*T + tau1*tau1 + tau2*tau2 + pow(E,inf_val*(tau1-600)) + pow(E,inf_val*(tau2-600)) + pow(E,-inf_val*(tau1+600)) + pow(E,-inf_val*(tau2+600)));
             cout << "cost function = sq_torque + exponential penalty " << endl;
  }

  f << dot(q1) == qd1;
  f << dot(q2) == qd2;
  // f << dot(qd1) == -(20*tau1 - 20*tau2 + 10*qd1*qd1*sin(q2) + 10*qd2*qd2*sin(q2) + 2*qd1*qd1*sin(2*q2) - 8*tau2*cos(q2) + 20*qd1*qd2*sin(q2))/(4*cos(q2)*cos(q2) - 45);
  // f << dot(qd2) == (20*tau1 - 56*tau2 + 28*qd1*qd1*sin(q2) + 10*qd2*qd2*sin(q2) + 4*qd1*qd1*sin(2*q2) + 2*qd2*qd2*sin(2*q2) + 8*tau1*cos(q2) - 16*tau2*cos(q2) + 20*qd1*qd2*sin(q2) + 4*qd1*qd2*sin(2*q2))/(2*cos(2*q2) - 43);
  f << dot(qd1) == -(48*tau1 - 48*tau2 + 24*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 18*qd1*qd1*sin(2*q2) - 72*tau2*cos(q2) + 48*qd1*qd2*sin(q2))/(36*cos(q2)*cos(q2) - 64);
  f << dot(qd2) == (48*tau1 - 240*tau2 + 120*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 36*qd1*qd1*sin(2*q2) + 18*qd2*qd2*sin(2*q2) + 72*tau1*cos(q2) - 144*tau2*cos(q2) + 48*qd1*qd2*sin(q2) + 36*qd1*qd2*sin(2*q2))/(18*cos(2*q2) - 46);

  ocp.subjectTo(f);
  ocp.subjectTo(AT_START, q1 == bounds[0] );
  ocp.subjectTo(AT_START, q2 == bounds[1]);
  ocp.subjectTo(AT_START, qd1 == bounds[2]);
  ocp.subjectTo(AT_START, qd2 == bounds[3]);

  ocp.subjectTo(AT_END, q1 == bounds[4]);
  ocp.subjectTo(AT_END, q2 == bounds[5]);
  ocp.subjectTo(AT_END, qd1 == bounds[6]);
  ocp.subjectTo(AT_END, qd2 == bounds[7]);

  // ocp.subjectTo(0.0 <= q1 <= 2*pi);
  // ocp.subjectTo(0.0 <= q2 <= 2*pi);
  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm
  algorithm.set(PRINT_COPYRIGHT,BT_FALSE);
  algorithm.set(PRINTLEVEL,LOW);
  algorithm.set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  algorithm.set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);

  algorithm.solve();                        // solves the problem.

  VariablesGrid grid;
  algorithm.getDifferentialStates(grid);
  float objVal = algorithm.getObjectiveValue();
  DVector final_state(4), req_state(4), diff(4);
  final_state = grid.getLastVector();
  req_state(0)=bounds[4];
  req_state(1)=bounds[5];
  req_state(2)=bounds[6];
  req_state(3)=bounds[7];
  diff = final_state - req_state;

  // if ((diff.getNorm(VN_L2) < 0.001)&&(objVal>=0.0)){
  if (diff.getNorm(VN_L2) < 0.001){
    flag = true;
    states_nm = "states_"+std::to_string(index)+".txt";
    parameters_nm = "parameters_"+std::to_string(index)+".txt";
    control_nm = "control_"+std::to_string(index)+".txt";
    algorithm.getDifferentialStates(states_nm.c_str());
    algorithm.getObjectiveValue(parameters_nm.c_str());
    algorithm.getControls(control_nm.c_str());
  }
  clearAllStaticCounters();
  return flag;
}

float rand_gen(float lim[2]) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = lim[1] - lim[0];
    float r = random * diff;
    return lim[0] + r;
}

int main(int argc, const char * argv[]){
  srand (time(NULL));
  int i;
  float q_lims[2] = {0.0,2*pi};
  float qd_lims[2] = {-30.0,30.0};

  if(argc!=3){
    cout << "Incorrect number of arguments." << endl;
  }
  else{
    int num_iter = atoi(argv[2]);
    i = 0;
    while(i<num_iter){
      float bounds[8] = {rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims),rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims)};
      cout << i << endl;
      cout << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << " " << bounds[6] << " " << bounds[7] << endl;
      bool success = gen_data(bounds,i,*argv[1]);
      if(success){
        i++;
      }
    }
  }
  return 0;
}
