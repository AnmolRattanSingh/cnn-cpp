#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <tuple>
#include <vector>

class Convolution {
   public:
    int depth;
    int in_depth;
    int in_height;
    int in_width;
    Eigen::Tensor<float, 3> input;
    Eigen::Tensor<float, 3> output;

    Eigen::Tensor<float, 4> kernels;
    Eigen::Tensor<float, 3> biases;

    Convolution(std::tuple<int, int, int> in_shape, int kernel_size, int depth);

    Eigen::Tensor<float, 3> forward(Eigen::Tensor<float, 3> input);

    Eigen::MatrixXd backward(Eigen::MatrixXd output_grad, int learning_rate);
};
