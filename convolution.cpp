#include "convolution.hpp"

#include <iostream>
#include <string>

using namespace std;

Convolution::Convolution(tuple<int, int, int> in_shape, int kernel_size, int depth) {
    in_depth = get<0>(in_shape);
    in_height = get<1>(in_shape);
    in_width = get<2>(in_shape);

    kernels.resize(depth, in_depth, kernel_size, kernel_size);
    kernels.setRandom();
    biases.resize(depth, in_height - kernel_size + 1, in_width - kernel_size + 1);
    biases.setRandom();
}

Eigen::Tensor<float, 3> Convolution::forward(Eigen::Tensor<float, 3> input) {
    this->input = input;

    this->output = this->biases;

    return this->output;
}
