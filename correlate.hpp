#pragma once

#include <eigen3/Eigen/Dense>

Eigen::MatrixXd correlate2d(Eigen::MatrixXd input_array, Eigen::MatrixXd kernel, int mode);

Eigen::MatrixXd convolve2d(Eigen::MatrixXd input_array, Eigen::MatrixXd kernel, int mode);
