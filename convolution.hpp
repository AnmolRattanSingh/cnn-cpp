#pragma once

#include <tuple>
#include <vector>
#include <eigen3/Eigen/Dense>

class Convolution {
public:
    int depth;
    int in_depth;
    int in_height;
    int in_width;
    std::tuple<int, int, int> output_shape;
    std::tuple<int, int, int, int> kernels_shape;
    Eigen::Matrix3d kernels;
    Eigen::Matrix4d biases;


    Convolution(std::tuple<int, int, int> in_shape, int kernel_size, int depth);

    Eigen::MatrixXd forward(Eigen::MatrixXd input);

    Eigen::MatrixXd backward(Eigen::MatrixXd output_grad, int learning_rate);
};
