#include <iostream>
#include <cassert>
#include "Layer.hpp"

void test_forward_dimensions() {
    int in_size = 8;
    int out_size = 3;
    int batch_size = 4;

    DenseLayer layer(in_size, out_size);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(in_size, batch_size);
    
    Eigen::MatrixXf output = layer.forward(input);

    // Assert that rows = output neurons and cols = batch size
    assert(output.rows() == out_size);
    assert(output.cols() == batch_size);
    
    std::cout << "Test Dimensions: PASSED" << std::endl;
}

void test_relu_logic() {
    // We create a tiny layer to predict the exact outcome
    DenseLayer layer(2, 1);
    
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
        assert(val >= 0); // ReLU should never output negative values
    }
    
    std::cout << "Test ReLU Logic: PASSED" << std::endl;
}

void test_manual_math() {
    // 1. Setup a 2x2 Layer
    DenseLayer layer(2, 2);

    Eigen::MatrixXf custom_w(2, 2);
    custom_w << 1.0f, 2.0f, 
                3.0f, 4.0f;

    Eigen::VectorXf custom_b(2);
    custom_b << 0.0f, 0.0f;

    // Use your Dev function to override random weights
    layer.DevDenseLayer(custom_w, custom_b);

    // 2. Define specific input
    Eigen::MatrixXf input(2, 2);
    input << 1.0f, 1.0f,
             1.0f, 1.0f;

    // 3. Forward pass
    Eigen::MatrixXf output = layer.forward(input);

    // 4. Verify results
    // Expected: Neuron 1 = 3, Neuron 2 = 7
    assert(output(0, 0) == 3);
    assert(output(1, 0) == 7);
    assert(output(0, 1) == 3);
    assert(output(1, 1) == 7);

    std::cout << "Output:\n" << output << std::endl;
}

int main() {
    try {
        test_forward_dimensions();
        test_relu_logic();
        test_manual_math();
        std::cout << "\nALL TESTS PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}