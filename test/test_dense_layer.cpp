#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <Eigen/Dense>
#include <Loss.hpp>
#include <Layer.hpp>
#include <Optimizer.hpp>

// Utility to compare floats with a small epsilon
bool is_near(float a, float b, float epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// Helper to create a test environment
struct TestEnv {
    std::shared_ptr<AdamOptimizer> opt;
    TestEnv() {
        opt = std::make_shared<AdamOptimizer>(0.1f); // High LR for tests
    }
};

// 1. Test Constructor and Initial Dimensions
void test_initialization() {
    TestEnv env;
    int in = 10, out = 5;
    DenseLayer layer(in, out, env.opt, "test_layer", act_type::ReLU);
    
    assert(layer.getWeights().rows() == out);
    assert(layer.getWeights().cols() == in);
    assert(layer.getBiases().size() == out);
    
    // Gradients should be initialized to zero (or match weight size)
    assert(layer.getWeightGrads().rows() == out);
    std::cout << "Test Initialization: PASSED" << std::endl;
}

// 2. Test Forward Pass (Linear + Activation)
void test_forward_logic() {
    TestEnv env;
    DenseLayer layer(2, 1, env.opt, "test_layer", act_type::ReLU);
    
    Eigen::MatrixXf w(1, 2); w << 2.0f, -1.0f;
    Eigen::VectorXf b(1); b << 5.0f;
    layer.setParameters(w, b); // Replaces DevDenseLayer

    Eigen::MatrixXf input(2, 1); input << 1.0f, 3.0f;
    
    // Calculation: (2*1 + -1*3) + 5 = 4.0 -> ReLU(4.0) = 4.0
    Eigen::MatrixXf output = layer.forward(input);
    assert(is_near(output(0, 0), 4.0f));

    // Test ReLU clipping
    input << -10.0f, 0.0f; 
    // Calculation: (2*-10 + -1*0) + 5 = -15.0 -> ReLU(-15.0) = 0.0
    output = layer.forward(input);
    assert(output(0, 0) == 0.0f);

    std::cout << "Test Forward Logic: PASSED" << std::endl;
}

// 3. Test Softmax Numerical Stability
void test_softmax_stability() {
    TestEnv env;
    DenseLayer layer(2, 2, env.opt, "test_layer", act_type::Softmax);
    
    Eigen::MatrixXf large_input(2, 1);
    large_input << 1000.0f, 999.0f; 

    Eigen::MatrixXf output = layer.forward(large_input);
    
    assert(!std::isnan(output(0, 0)));
    assert(is_near(output.col(0).sum(), 1.0f));
    
    std::cout << "Test Softmax Stability: PASSED" << std::endl;
}

// 4. Test Backward Pass (Gradient Calculation)
void test_backward_gradients() {
    TestEnv env;
    int in = 2, out = 1;
    DenseLayer layer(in, out, env.opt, "test_layer", act_type::None); 
    
    Eigen::MatrixXf w(1, 2); w << 0.5f, 0.5f;
    Eigen::VectorXf b(1); b << 0.0f;
    layer.setParameters(w, b);

    Eigen::MatrixXf input(2, 1); input << 2.0f, 4.0f;
    layer.forward(input); 

    Eigen::MatrixXf grad_out(1, 1); grad_out << 1.0f;
    layer.backward(grad_out);

    // dL/dW = grad_out * input^T = [1.0] * [2.0, 4.0] = [2.0, 4.0]
    assert(is_near(layer.getWeightGrads()(0, 0), 2.0f));
    assert(is_near(layer.getWeightGrads()(0, 1), 4.0f));
    assert(is_near(layer.getBiasGrads()(0), 1.0f));

    std::cout << "Test Backward Gradients: PASSED" << std::endl;
}

// 5. Test Adam Optimizer Update
void test_adam_update() {
    TestEnv env;
    DenseLayer layer(2, 1, env.opt, "test_layer", act_type::None);
    
    Eigen::MatrixXf w(1, 2); w << 1.0f, 1.0f;
    Eigen::VectorXf b(1); b << 1.0f;
    layer.setParameters(w, b);

    // Inject fake gradients
    Eigen::MatrixXf wg(1, 2); wg << 1.0f, 1.0f;
    Eigen::VectorXf bg(1); bg << 1.0f;
    layer.setGrads(wg, bg);

    // Update sequence
    layer.update(); 
    env.opt->step(); // Adam requires a time-step increment to work

    // Weight should have decreased
    assert(layer.getWeights()(0, 0) < 1.0f);
    
    std::cout << "Test Adam Update: PASSED" << std::endl;
}

int main() {
    std::cout << "--- STARTING REFACTORED DENSE LAYER TESTS ---" << std::endl;
    
    test_initialization();
    test_forward_logic();
    test_softmax_stability();
    test_backward_gradients();
    test_adam_update();

    std::cout << "\nALL UNIT TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}