#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <acado_optimal_control.hpp>

#define pi 3.14159

int main(){

  USING_NAMESPACE_ACADO

  DifferentialState        q1,q2,qd1,qd2;   // the differential states
  Control                  tau1,tau2;       // the control input u
  Parameter                T;
  DifferentialEquation     f;

  //  -------------------------------------
  OCP ocp( 0.0, 0.5 );
  ocp.minimizeMayerTerm( T*T + tau1*tau1 + tau2*tau2 );

  f << dot(q1) == qd1;
  f << dot(q2) == qd2;
  f << dot(qd1) == -(48*tau1 - 48*tau2 + 24*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 18*qd1*qd1*sin(2*q2) - 72*tau2*cos(q2) + 48*qd1*qd2*sin(q2))/(36*cos(q2)*cos(q2) - 64);
  f << dot(qd2) == (48*tau1 - 240*tau2 + 120*qd1*qd1*sin(q2) + 24*qd2*qd2*sin(q2) + 36*qd1*qd1*sin(2*q2) + 18*qd2*qd2*sin(2*q2) + 72*tau1*cos(q2) - 144*tau2*cos(q2) + 48*qd1*qd2*sin(q2) + 36*qd1*qd2*sin(2*q2))/(18*cos(2*q2) - 46);

  ocp.subjectTo(f);
  ocp.subjectTo(AT_START, q1 ==  0.0 );
  ocp.subjectTo(AT_START, q2 ==  0.0 );
  ocp.subjectTo(AT_START, qd1 ==  0.0 );
  ocp.subjectTo(AT_START, qd2 ==  0.0 );

  ocp.subjectTo(AT_END, q1 == 4*pi/2.0);
  ocp.subjectTo(AT_END, q2 == 4*pi/2.0);
  ocp.subjectTo(AT_END, qd1 == 0.0);
  ocp.subjectTo(AT_END, qd2 == 0.0);

  // ocp.subjectTo(0.0<= q1 <= 2*pi);
  // ocp.subjectTo(0.0 <= q2 <= 2*pi);
  ocp.subjectTo(-300 <= tau1 <= 300);  // bounds on the control input u,
  ocp.subjectTo(-300 <= tau2 <= 300);
  // ocp.subjectTo(-25.0<= qd1 <= 25.0);
  // ocp.subjectTo(-25.0 <= qd2 <= 25.0);

  // ocp.subjectTo(0.1 <= T <= 15.0);     // and the time horizon T.
  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);
  algorithm.set(PRINT_COPYRIGHT,BT_FALSE);
  algorithm.set(PRINTLEVEL,LOW);
  // algorithm.set(PRINT_INTEGRATOR_PROFILE,BT_FALSE);
  // algorithm.set(PRINT_SCP_METHOD_PROFILE,BT_FALSE);

  GnuplotWindow window(PLOT_AT_END);
  window.addSubplot(q1, "q1");
  window.addSubplot(q2, "q2");
  window.addSubplot(qd1, "qd1");
  window.addSubplot(qd2, "qd2");
  window.addSubplot(tau1, "tau1");
  window.addSubplot(tau2, "tau2");

      // the optimization algorithm
  algorithm << window;
  LogRecord logRecord(LOG_AT_END);
  // logRecord << LOG_CONTROLS;
  logRecord << LOG_OBJECTIVE_VALUE;
  // logRecord << LOG_INTERMEDIATE_STATES;


  algorithm << logRecord;

  // solve the optimization problem
  algorithm.solve( );
  algorithm.getControls("control_log.txt");
  // algorithm.getObjectiveValue("ob_log.txt");
  VariablesGrid asd;
  algorithm.getDifferentialStates(asd);
  DVector ok = asd.getLastVector();
  DVector not_ok(4);
  not_ok.setZero();
  ok.setZero();
  if (ok != not_ok){
    std::cout << "Works" <<std::endl;
  }

    // algorithm.getAlgebraicStates("alstate_test.txt");
  algorithm.getDifferentialStates("diffstate_log.txt");
  // algorithm.printInfo();

  // get the logging object back and print it
  algorithm.getLogRecord(logRecord);
  logRecord.print();


  return 0;
}
