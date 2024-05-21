#include "convolution.hpp"
#include <iostream>
#include <H5Cpp.h>
using namespace std;
using namespace H5;

int main(void) {
    tuple<int, int, int> test = make_tuple(1,1,1);
    Convolution* test_conv = new Convolution(test, 5, 10);
    cout << test_conv->kernels << "\n" << endl;
    cout << test_conv->biases << endl;
    test_conv->forward(test_conv->biases);

    // Open an existing HDF5 file
    H5File file("../MNISTdata.hdf5", H5F_ACC_RDONLY);

    // Open the dataset named "dataset_name" in the file
    DataSet dataset = file.openDataSet("x_train");

    // Get the datatype and dataspace of the dataset
    DataType dtype = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();

    // Get the number of dimensions in the dataset
    int rank = dataspace.getSimpleExtentNdims();

    // Get the size of each dimension
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data(), nullptr);

    // Ensure the dataset is 2D
    if (rank != 2) {
        throw std::runtime_error("Expected a 2D dataset");
    }

    // Allocate an Eigen matrix to read the dataset
    Eigen::MatrixXf data(dims[0], dims[1]);


    dataset.read(data.data(), PredType::NATIVE_FLOAT);

    cout << data << endl;
}
