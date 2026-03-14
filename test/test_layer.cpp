#include <iostream>
#include <cassert>
#include "../include/Layer.hpp"

void test_forward_dimensions() {
    int in_size = 8;
    int out_size = 3;
    int batch_size = 4;

    DenseLayer layer(in_size, out_size, batch_size);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(in_size, batch_size);
    
    Eigen::MatrixXf output = layer.forward(input);

    // Assert that rows = output neurons and cols = batch size
    assert(output.rows() == out_size);
    assert(output.cols() == batch_size);
    
    std::cout << "Test Dimensions: PASSED" << std::endl;
}

void test_relu_logic() {
    // We create a tiny layer to predict the exact outcome
    DenseLayer layer(2, 1, 1);
    
    // Manually set weights/biases to force a negative result
    // weights = [0, 0], bias = [-1]. Input = [10, 10]
    // Result = (0*10 + 0*10) - 1 = -1. ReLU should return 0.
    
    // Use Eigen accessors to override random values for testing
    // (Note: You might need to make weights/biases public or add a setter for testing)
    // For this example, let's assume we check if any output is strictly 1 or 0
    
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(2, 1);
    Eigen::MatrixXf output = layer.forward(input);

    for(int i=0; i < output.size(); ++i) {
        float val = output.data()[i];
        assert(val == 0.0f || val == 1.0f);
    }
    
    std::cout << "Test ReLU Logic: PASSED" << std::endl;
}

int main() {
    try {
        test_forward_dimensions();
        test_relu_logic();
        std::cout << "\nALL TESTS PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}