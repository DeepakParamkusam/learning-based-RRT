#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>

#define pi 3.14159

int main(){

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
  ocp.subjectTo(AT_START, q1 ==  0.0 );
  ocp.subjectTo(AT_START, q2 ==  0.0 );
  ocp.subjectTo(AT_START, qd1 ==  0.0 );
  ocp.subjectTo(AT_START, qd2 ==  0.0 );

  ocp.subjectTo(AT_END, q1 == pi/2.0);
  ocp.subjectTo(AT_END, q2 == pi/2.0);
  ocp.subjectTo(AT_END, qd1 == 0.0);
  ocp.subjectTo(AT_END, qd2 == 0.0);

  ocp.subjectTo(-100 <= tau1 <= 100);  // bounds on the control input u,
  ocp.subjectTo(-100 <= tau2 <= 100);
  ocp.subjectTo(0.1 <= T <= 15.0);     // and the time horizon T.
  //  -------------------------------------

  GnuplotWindow window;
  window.addSubplot(q1, "q1");
  window.addSubplot(q2, "q2");
  window.addSubplot(qd1, "qd1");
  window.addSubplot(qd2, "qd2");
  window.addSubplot(tau1, "tau1");
  window.addSubplot(tau2, "tau2");

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm
  algorithm << window;
  algorithm.solve();                        // solves the problem.


  return 0;
}
