#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#define pi 3.14159

using namespace std;

static boost::random::mt19937 gen;

int test(float bounds[8]);
float rand_gen(float lim[2]);

int test(float bounds[8]){

  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f( 0.0, 0.5 );

  //  -------------------------------------
  OCP ocp( 0.0, 0.5 );
  
  ocp.minimizeLagrangeTerm(tau1*tau1 + tau2*tau2);
  // ocp.minimizeMayerTerm(T*T + tau1*tau1 + tau2*tau2);

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

  ocp.subjectTo(-400 <= tau1 <= 400);  // bounds on the control input u,
  ocp.subjectTo(-400 <= tau2 <= 400);
  //  -------------------------------------

  // ---------PLOT FOR TESTING
  // GnuplotWindow window;
  // window.addSubplot(q1, "q1");
  // window.addSubplot(q2, "q2");
  // window.addSubplot(qd1, "qd1");
  // window.addSubplot(qd2, "qd2");
  // window.addSubplot(tau1, "tau1");
  // window.addSubplot(tau2, "tau2");

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm
  algorithm.set( DISCRETIZATION_TYPE , MULTIPLE_SHOOTING);
  algorithm.set( INTEGRATOR_TYPE , INT_RK45);
  algorithm.set( HESSIAN_APPROXIMATION   , BLOCK_BFGS_UPDATE);
  algorithm.set( KKT_TOLERANCE   , 1e-4); 
  algorithm.set( ABSOLUTE_TOLERANCE, 1e-4);
  algorithm.set( INTEGRATOR_TOLERANCE, 1e-4);
  algorithm.set( MAX_NUM_ITERATIONS, 1000);
  algorithm.set( MAX_NUM_INTEGRATOR_STEPS, 10000);

  algorithm.set(PRINT_COPYRIGHT,BT_FALSE);
  algorithm.set(PRINTLEVEL,LOW);
  algorithm.set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  algorithm.set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);
  // algorithm << window;
  algorithm.solve();                        // solves the problem.

  algorithm.getDifferentialStates("states.txt");
  algorithm.getObjectiveValue("parameters.txt");
  algorithm.getControls("control.txt");

  clearAllStaticCounters();
  return 0;
}

float rand_gen(float lim[2]) {
    static boost::random::uniform_real_distribution<> dist(0, 1);
    return lim[0] + dist(gen)*(lim[1] - lim[0]);
}

int main(int argc, char const *argv[])
{
  float q_lims[2] = {0.0,2*pi};
  float qd_lims[2] = {-30.0,30.0};
  float bounds[8] = {5.11906, 0.851226, 24.3475, 20.1005, 0.797881, 6.08757, 24.8026, -16.738};
  // float bounds[8] = {rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims),rand_gen(q_lims),rand_gen(q_lims),rand_gen(qd_lims),rand_gen(qd_lims)};
  cout << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << " " << bounds[6] << " " << bounds[7] << endl;
  bool success = test(bounds);
  return 0;
}
