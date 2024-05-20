#include "convolution.hpp"

#include <iostream>
#include <string>

using namespace std;

Convolution::Convolution(tuple<int, int, int> in_shape, int kernel_size, int depth){
    in_depth = get<0>(in_shape);
    in_height = get<1>(in_shape);
    in_width = get<2>(in_shape);
    output_shape = make_tuple(depth, in_height - kernel_size + 1, in_width - kernel_size + 1);
    kernels_shape = make_tuple(depth, in_depth, kernel_size, kernel_size);

    kernels = Eigen::Matrix3d::Random();
    biases = Eigen::Matrix4d::Random();
}

