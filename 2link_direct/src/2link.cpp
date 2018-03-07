#include <iostream>
#include <random>
#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#define pi 3.14159

using namespace std;

static boost::random::mt19937 gen;

int gen_data(float bounds[8], int index, char thread);
float rand_gen(float lim[2]);

float rand_gen(float lim[2]){
  static boost::random::uniform_real_distribution<> dist(0, 1);
  return lim[0] + dist(gen)*(lim[1] - lim[0]);
}

int gen_data(float bounds[8], int index, char thread){
  string states_nm, parameters_nm, control_nm;
  bool flag = false;

  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f( 0.0, 0.5 );

  //  -------------------------------------

  OCP ocp( 0.0, 0.5,5 );  
  ocp.minimizeLagrangeTerm(1+tau1*tau1 + tau2*tau2);
  //ocp.minimizeMayerTerm(q1*q1 + q2*q2 + qd1*qd1 + qd2*qd2 );

  f << dot(q1) == qd1;
  f << dot(q2) == qd2;
  f << dot(qd1) == -(48*tau1 - 48*tau2 + 24*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 18*qd1*qd1*sin(2*q2) - 72*tau2*cos(q2) + 48*qd1*qd2*sin(q2))/(36*cos(q2)*cos(q2) - 64);
  f << dot(qd2) == (48*tau1 - 240*tau2 + 120*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 36*qd1*qd1*sin(2*q2) + 18*qd2*qd2*sin(2*q2) + 72*tau1*cos(q2) - 144*tau2*cos(q2) + 48*qd1*qd2*sin(q2) + 36*qd1*qd2*sin(2*q2))/(18*cos(2*q2) - 46);

  ocp.subjectTo(f);
  ocp.subjectTo(AT_START, q1 ==  bounds[0] );       

  ocp.subjectTo(AT_START, q2 ==  bounds[1] );       

  ocp.subjectTo(AT_START, qd1 ==  bounds[2] );
  ocp.subjectTo(AT_START, qd2 ==  bounds[3] );

  ocp.subjectTo(AT_END, q1 == bounds[4]);
  ocp.subjectTo(AT_END, q2 == bounds[5]);
  ocp.subjectTo(AT_END, qd1 == bounds[6]);
  ocp.subjectTo(AT_END, qd2 == bounds[7]);

  // bounds on the control input u
  ocp.subjectTo(-400 <= tau1 <= 400);  
  ocp.subjectTo(-400 <= tau2 <= 400);

  //  -------------------------------------
  
  // Optimization algorithm
  OptimizationAlgorithm algorithm(ocp);     
  algorithm.set( DISCRETIZATION_TYPE , SINGLE_SHOOTING);
  algorithm.set( INTEGRATOR_TYPE , INT_RK45);
  algorithm.set( HESSIAN_APPROXIMATION   , BLOCK_BFGS_UPDATE);
  algorithm.set( KKT_TOLERANCE   , 1e-1); 
  algorithm.set( ABSOLUTE_TOLERANCE, 1e-1);
  algorithm.set( INTEGRATOR_TOLERANCE, 1e-1);
  algorithm.set( MAX_NUM_ITERATIONS, 1000);
  algorithm.set( MAX_NUM_INTEGRATOR_STEPS, 10000);

  algorithm.set(PRINT_COPYRIGHT,BT_FALSE);
  algorithm.set(PRINTLEVEL,LOW);
  algorithm.set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  algorithm.set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);
  bool return_code = algorithm.solve();                        

  // VariablesGrid grid;
  // algorithm.getDifferentialStates(grid);
  // DVector final_state(4), diff(4);
  // final_state = grid.getLastVector(); 
  
  // diff(0) = final_state[0] - bounds[4];
  // diff(1) = final_state[1] - bounds[5];
  // diff(2) = final_state[2] - bounds[6];
  // diff(3) = final_state[3] - bounds[7];

  // if (diff.getNorm(VN_L2) < 0.001){
  
  if (return_code == 0){
    flag = true;
    // cout << thread;
    states_nm = "states_" + to_string(index) + "_" + thread + ".txt";
    parameters_nm = "parameters_" + to_string(index) + "_" + thread + ".txt";
    control_nm = "control_" + to_string(index) + "_" + thread + ".txt";
    algorithm.getDifferentialStates(states_nm.c_str());
    algorithm.getObjectiveValue(parameters_nm.c_str());
    algorithm.getControls(control_nm.c_str());
  }

  clearAllStaticCounters();
  return flag;
}

int main(int argc, char const *argv[])
{  
  float q_lims[2] = {0.0,2*pi};
  float qd_lims[2] = {-30.0,30.0};
    
  if(argc != 3){
    cout << "Incorrect number of arguments." << endl;
  }
  else{
    int num_iter = atoi(argv[1]);
    
    gen.seed(time(0) + int(*argv[2]));
    int i = 0;
    while(i < num_iter){
      float bounds[8] = {rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims),rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims)};
      cout << i << endl;
      cout << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << " " << bounds[6] << " " << bounds[7] << endl;
      bool success = gen_data(bounds, i, *argv[2]);
      if(success){
        i++;
      }
    }
  }
  return 0;
}

