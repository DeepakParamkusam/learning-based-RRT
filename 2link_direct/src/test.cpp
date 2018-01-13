#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#define pi 3.14159

using namespace std;

int main(){
  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f( 0.0, 0.5 );

  //  -------------------------------------
  OCP* ocp;
  OptimizationAlgorithm* algorithm;
  
  ocp = new OCP(0.0,0.5);
  ocp->minimizeLagrangeTerm(tau1*tau1 + tau2*tau2);
  ocp->minimizeMayerTerm(q1*q1 + q2*q2 + qd1*qd1 + qd2*qd2);

  f << dot(q1) == qd1;
  f << dot(q2) == qd2;
  f << dot(qd1) == -(48*tau1 - 48*tau2 + 24*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 18*qd1*qd1*sin(2*q2) - 72*tau2*cos(q2) + 48*qd1*qd2*sin(q2))/(36*cos(q2)*cos(q2) - 64);
  f << dot(qd2) == (48*tau1 - 240*tau2 + 120*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 36*qd1*qd1*sin(2*q2) + 18*qd2*qd2*sin(2*q2) + 72*tau1*cos(q2) - 144*tau2*cos(q2) + 48*qd1*qd2*sin(q2) + 36*qd1*qd2*sin(2*q2))/(18*cos(2*q2) - 46);

   //  -------------------------------------
  ocp->subjectTo(f);
  ocp->subjectTo(-400 <= tau1 <= 400);  // bounds on the control input u,
  ocp->subjectTo(-400 <= tau2 <= 400);

  // ---------PLOT FOR TESTING
  GnuplotWindow window;
  window.addSubplot(q1, "q1");
  window.addSubplot(q2, "q2");
  window.addSubplot(qd1, "qd1");
  window.addSubplot(qd2, "qd2");
  window.addSubplot(tau1, "tau1");
  window.addSubplot(tau2, "tau2");
  
  // float bounds[8] = {0,0,0,0,pi,0,0,0};
  float bounds[8] = {1.3482846021652222e+00,  1.4300998449325562e+00,  -2.0796922683715820e+01, 2.4224172592163086e+01,  3.0019950866699219e+00,  4.3447545170783997e-01,  1.1785173416137695e+01,  -2.3085798263549805e+01};
  // float bounds[8] = {5.9785609245300293e+00, 3.9170422554016113e+00,  2.4079307556152344e+01,  -1.8390211105346680e+01, 6.2824505567539024e-01,  6.2820329666137180e+00,  8.4404172897345422e+00, -2.5410150527954766e+01};
  cout << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << " " << bounds[6] << " " << bounds[7] << endl;

  ocp->subjectTo(AT_START, q1 ==  bounds[0] );
  ocp->subjectTo(AT_START, q2 ==  bounds[1] );       
  ocp->subjectTo(AT_START, qd1 ==  bounds[2] );
  ocp->subjectTo(AT_START, qd2 ==  bounds[3] );

  ocp->subjectTo(AT_END, q1 == bounds[4]);
  ocp->subjectTo(AT_END, q2 == bounds[5]);
  ocp->subjectTo(AT_END, qd1 == bounds[6]);
  ocp->subjectTo(AT_END, qd2 == bounds[7]);

  // the optimization algorithm
  algorithm = new OptimizationAlgorithm(*ocp);
  algorithm->set( DISCRETIZATION_TYPE , MULTIPLE_SHOOTING);
  algorithm->set( INTEGRATOR_TYPE , INT_RK45);
  algorithm->set( HESSIAN_APPROXIMATION   , BLOCK_BFGS_UPDATE);
  algorithm->set( KKT_TOLERANCE   , 1e-4); 
  algorithm->set( ABSOLUTE_TOLERANCE, 1e-4);
  algorithm->set( INTEGRATOR_TOLERANCE, 1e-4);
  algorithm->set( MAX_NUM_ITERATIONS, 1000);
  algorithm->set( MAX_NUM_INTEGRATOR_STEPS, 10000);

  // algorithm->set(PRINT_COPYRIGHT,BT_FALSE);
  // algorithm->set(PRINTLEVEL,LOW);
  // algorithm->set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  // algorithm->set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);
  bool ok = algorithm->solve(); 
                         // solves the problem.
  *algorithm << window;
  if (ok == 0){
    cout << "SUCCESS!";
    algorithm->getDifferentialStates("states.txt");
    algorithm->getObjectiveValue("parameters.txt");
    algorithm->getControls("control.txt");
  }



  return 0;
}



