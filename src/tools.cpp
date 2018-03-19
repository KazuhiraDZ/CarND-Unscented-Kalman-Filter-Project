#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //check
  if (estimations.size() == 0)	{
  	  cout << "Input is empty " << endl;
  	  return rmse;
  }

  if (estimations.size() != ground_truth.size()) {
  	  cout << "Dimension is unvalid " << endl;
  	  return rmse;
  }

  // accumulate square
  for (int i=0; i< estimations.size(); i++){
  	  VectorXd residual = estimations[i] - ground_truth[i];
  	  // Coefficient-wise multiplication
  	  residual = residual.array() * residual.array();
  	  rmse += residual;
  }

  // accumulate square
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;

}