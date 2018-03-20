#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {

  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.9;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.9;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  n_x_ = x_.size();
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // weights for predicttion
  weights_ = VectorXd(2 * n_aug_ + 1);

  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
  0, std_radphi_ * std_radphi_, 0,
  0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2,2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
  0, std_laspy_ * std_laspy_;

  H_ = MatrixXd(2,n_x_);
  H_.fill(0.0);
  H_(0,0) = 1;
  H_(1,1) = 1;
}

UKF::~UKF() {}


void UKF::NormAng(double ang){
  while (ang > M_PI) ang -= 2. * M_PI;
  while (ang < -M_PI) ang += 2. * M_PI;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  // if not initialze, we should use the measurement value as the initial value
  if (!is_initialized_){
    // initial P
    P_ << 1,0,0,0,0,
    0,1,0,0,0,
    0,0,1,0,0,
    0,0,0,1,0,
    0,0,0,0,1;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
      float rho = meas_package.raw_measurements_[0]; // range
      float phi = meas_package.raw_measurements_[1]; // bearing
      float rho_dot = meas_package.raw_measurements_[2]; // velocity of rho
      // coordinates convertion from polar to cartesian
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      float v = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      // we don't know the velocity from the first measurement of the LASER, so use zeros
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      if (fabs(x_(0)) < 0.0001 and fabs(x_(1)) < 0.0001){
        x_(0) = 0.0001;
        x_(1) = 0.0001;
      }
    }

    // Initialize weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < weights_.size(); i++) {
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    // save the inital timestamp for dt calculation
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // calculate the timestamp between measurements in seconds
  double dt = meas_package.timestamp_ - time_us_;
  dt /= 1000000.0; // convert micros to s
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // Sigma points generation (add augment) 
  // augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  // augmented state covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
  // sigma point matrix
  MatrixXd Xsig = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  // fill the matrix
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // calculate square root of P
  MatrixXd L = P_aug.llt().matrixL();

  // set the first column of sigma point matrix
  Xsig.col(0) = x_aug;

  // set remaining sigma point
  for (int i=0; i<n_aug_; i++){
    Xsig.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // Sigma Prediction
  for (int i=0; i<2 * n_aug_ + 1; i++){
    double px = Xsig(0,i);
    double py = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double nu_a = Xsig(5,i);
    double nu_yawdd = Xsig(6,i);

    double px_p, py_p;
    // when yawd = 0, means straight
    if(fabs(yawd) > 0.001){
      px_p = px + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v/yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }else{
      px_p = px + v * cos(yaw) * delta_t;
      py_p = py + v * sin(yaw) * delta_t;
    }

    double v_p = v;
    double yaw_p = yawd + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    py_p += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
    v_p += nu_a * delta_t;
    yaw_p += 0.5 * delta_t * delta_t * nu_yawdd;
    yawd_p += nu_yawdd * delta_t;


    // write predicted sigma point
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

  }

  //Predicted state mean
  x_ = Xsig_pred_ * weights_; //vectorised sum

  P_.fill(0.0);
  // Iterate over sigma points
  for (int i=0; i<2 * n_aug_ + 1; i++){ 
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormAng(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);
  // for (int i=0; i<2 * n_aug_ + 1; i++){
  //   double px = Xsig_pred_(0,i);
  //   double py = Xsig_pred_(1,i);

  //   Zsig(0,i) = px;
  //   Zsig(1,i) = py;
  // }
  // UpdateUKF(meas_package, Zsig, 2);

  VectorXd z = meas_package.raw_measurements_;
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_lidar_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  x_ += K * y;
  P_ = (MatrixXd::Identity(n_x_,n_x_) - K * H_) * P_;
  NIS_laser_ = y.transpose() * S.inverse() * y;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

  for (int i=0; i<2 * n_aug_ + 1; i++){
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double vx = cos(yaw) * v;
    double vy = sin(yaw) * v;

    Zsig(0,i) = sqrt(px * px + py * py);
    Zsig(1,i) = atan2(py, px);
    Zsig(2,i) = (px * vx + py * vy) / sqrt(px * px + py * py);

  }

  UpdateUKF(meas_package, Zsig, 3);
}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){
  VectorXd z_pred = VectorXd(n_z);

  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i=0; i<2 * n_aug_ + 1; i++){

    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(z_diff(1));
    }
    S = S + weights_(i) * z_diff * z_diff.transpose();
  } 

  // add measurement noise
  MatrixXd R = MatrixXd(n_z,n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    R = R_radar_;
  }

  S = S + R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i=0; i< 2 * n_aug_ + 1; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(z_diff(1));
      NormAng(x_diff(3));
    }

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // measurement
  VectorXd z = meas_package.raw_measurements_;

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(z_diff(1));
  }

  // update
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}

