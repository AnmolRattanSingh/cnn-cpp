#include "convolution.hpp"
#include <iostream>
using namespace std;

int main(void) {
    tuple<int, int, int> test = make_tuple(1,1,1);
    Convolution* test_conv = new Convolution(test, 5, 10);
    cout << test_conv->kernels << "\n" << endl;
    cout << test_conv->biases << endl;
}
