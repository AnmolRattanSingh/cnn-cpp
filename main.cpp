#include <H5Cpp.h>

#include <iostream>

#include "convolution.hpp"
using namespace std;
using namespace H5;

int main(void) {
    tuple<int, int, int> test = make_tuple(1, 28, 28);
    Convolution* test_conv = new Convolution(test, 5, 10);
    cout << test_conv->biases << endl;
    cout << test_conv->kernels.size() << endl;
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

    // Ensure the dataset is 2D or more
    if (rank < 2) {
        throw std::runtime_error("Expected a 2D or higher-dimensional dataset");
    }

    // Allocate an Eigen tensor to read the dataset
    Eigen::Tensor<float, 2> data(dims[0], dims[1]);

    // Read the dataset into the Eigen tensor
    dataset.read(data.data(), PredType::NATIVE_FLOAT);

    // Print the dataset
    std::cout << "Dataset contents:\n"
              << data << std::endl;
}
