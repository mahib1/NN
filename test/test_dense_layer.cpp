#include <iostream>
#include <cassert>
#include <cmath>
#include <Loss.hpp>
#include <Layer.hpp>

// NOTE: These tests are written by AI.

// Utility to compare floats with a small epsilon
bool is_near(float a, float b, float epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// 1. Test Constructor and Initial Dimensions
void test_initialization() {
    int in = 10, out = 5;
    DenseLayer layer(in, out, Activation::ReLU);
    
    assert(layer.getWeights().rows() == out);
    assert(layer.getWeights().cols() == in);
    assert(layer.getBiases().size() == out);
    
    // Ensure Adam buffers are zeroed
    assert(layer.getWeightGrads().sum() == 0);
    std::cout << "Test Initialization: PASSED" << std::endl;
}

// 2. Test Forward Pass (Linear + Activation)
void test_forward_logic() {
    DenseLayer layer(2, 1, Activation::ReLU);
    
    Eigen::MatrixXf w(1, 2); w << 2.0f, -1.0f;
    Eigen::VectorXf b(1); b << 5.0f;
    layer.DevDenseLayer(w, b);

    Eigen::MatrixXf input(2, 1); input << 1.0f, 3.0f;
    
    // Calculation: (2*1 + -1*3) + 5 = 4.0
    // ReLU(4.0) = 4.0
    Eigen::MatrixXf output = layer.forward(input);
    assert(is_near(output(0, 0), 4.0f));

    // Test ReLU clipping
    input << -10.0f, 0.0f; 
    // Calculation: (2*-10 + -1*0) + 5 = -15.0
    // ReLU(-15.0) = 0.0
    output = layer.forward(input);
    assert(output(0, 0) == 0.0f);

    std::cout << "Test Forward Logic: PASSED" << std::endl;
}

// 3. Test Softmax Numerical Stability
void test_softmax_stability() {
    DenseLayer layer(2, 2, Activation::Softmax);
    
    // Extremely large values that would normally cause exp() to overflow (NaN)
    Eigen::MatrixXf large_input(2, 1);
    large_input << 1000.0f, 999.0f; 

    Eigen::MatrixXf output = layer.forward(large_input);
    
    // Result should not be NaN
    assert(!std::isnan(output(0, 0)));
    // Sum of probabilities must be 1.0
    assert(is_near(output.col(0).sum(), 1.0f));
    
    std::cout << "Test Softmax Stability: PASSED" << std::endl;
}

// 4. Test Backward Pass (Gradient Calculation)
void test_backward_gradients() {
    int in = 2, out = 1, batch = 1;
    DenseLayer layer(in, out, Activation::None); // Keep it linear for easy checking
    
    Eigen::MatrixXf w(1, 2); w << 0.5f, 0.5f;
    Eigen::VectorXf b(1); b << 0.0f;
    layer.DevDenseLayer(w, b);

    Eigen::MatrixXf input(2, 1); input << 2.0f, 4.0f;
    layer.forward(input); // Cache input and z

    // Assume the next layer sent back a gradient of 1.0
    Eigen::MatrixXf grad_out(1, 1); grad_out << 1.0f;
    
    // Run backward
    layer.backward(grad_out);

    // dL/dW = grad_out * input^T = [1.0] * [2.0, 4.0] = [2.0, 4.0]
    assert(is_near(layer.getWeightGrads()(0, 0), 2.0f));
    assert(is_near(layer.getWeightGrads()(0, 1), 4.0f));
    
    // dL/dB = sum(grad_out) = 1.0
    assert(is_near(layer.getBiasGrads()(0), 1.0f));

    std::cout << "Test Backward Gradients: PASSED" << std::endl;
}

// 5. Test Adam Optimizer Update
void test_adam_update() {
    DenseLayer layer(2, 1, Activation::None);
    
    // Set weights to 1.0
    Eigen::MatrixXf w(1, 2); w << 1.0f, 1.0f;
    Eigen::VectorXf b(1); b << 1.0f;
    layer.DevDenseLayer(w, b);

    // Inject a fake gradient of 1.0
    Eigen::MatrixXf wg(1, 2); wg << 1.0f, 1.0f;
    Eigen::VectorXf bg(1); bg << 1.0f;
    layer.setWeightGrads(wg);
    layer.setBiasGrads(bg);

    // Update with a large learning rate for visibility
    layer.update(0.1f);

    // Weight should have decreased (moving opposite to gradient)
    assert(layer.getWeights()(0, 0) < 1.0f);
    
    std::cout << "Test Adam Update: PASSED" << std::endl;
}

void test_softmax_learning() {
    // 1. Setup a layer that should predict "Class 1" (index 0)
    DenseLayer layer(2, 2, Activation::Softmax);
    
    // 2. Fake some data: Input [1, 1], Target is Class 1: [1, 0]
    Eigen::MatrixXf input(2, 1); input << 1.0f, 1.0f;
    Eigen::MatrixXf y_true(2, 1); y_true << 1.0f, 0.0f;
    
    // 3. Forward Pass
    Eigen::MatrixXf y_pred = layer.forward(input);
    
    // 4. Manual "Loss Gradient" (The shortcut!)
    Eigen::MatrixXf loss_grad = y_pred - y_true;
    
    // 5. Backward + Update
    layer.backward(loss_grad);
    float initial_weight = layer.getWeights()(0, 0);
    layer.update(0.1f); // Use a high LR to see movement
    
    // 6. Verification: Did the weight move?
    assert(layer.getWeights()(0, 0) != initial_weight);
    
    std::cout << "Test Softmax Learning: PASSED" << std::endl;
}

void test_full_training_step() {
    // 1. Setup a simple 2-input, 2-output classifier
    DenseLayer layer(2, 2, Activation::Softmax);
    
    // 2. Data: Input [1, 0], Target is Class 1: [1, 0]
    Eigen::MatrixXf input(2, 1); input << 1.0f, 0.0f;
    Eigen::MatrixXf y_true(2, 1); y_true << 1.0f, 0.0f;
    
    // 3. First Pass: Calculate initial loss
    Eigen::MatrixXf y_pred_initial = layer.forward(input);
    float loss_initial = compute_cross_entropy(y_pred_initial, y_true);
    
    // 4. Backprop: (Pred - True) is the gradient for Softmax
    Eigen::MatrixXf grad = y_pred_initial - y_true;
    layer.backward(grad);
    layer.update(0.1f); // Update weights
    
    // 5. Second Pass: Loss should be lower now
    Eigen::MatrixXf y_pred_new = layer.forward(input);
    float loss_new = compute_cross_entropy(y_pred_new, y_true);
    
    std::cout << "Initial Loss: " << loss_initial << " | New Loss: " << loss_new << std::endl;
    
    // If the network is learning, loss_new must be less than loss_initial
    assert(loss_new < loss_initial);
    
    std::cout << "Test Full Training Step: PASSED" << std::endl;
}

int main() {
    std::cout << "--- STARTING DENSE LAYER TESTS ---" << std::endl;
    
    test_initialization();
    test_forward_logic();
    test_softmax_stability();
    test_backward_gradients();
    test_adam_update();
    test_softmax_learning();
    test_full_training_step();

    std::cout << "\nALL UNIT TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}