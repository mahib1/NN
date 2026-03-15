#include <Layer.hpp>
#include <cmath>
#include <iostream>

Eigen::MatrixXf Layer::forward(const Eigen::MatrixXf& input) {
    return input; //Placeholder
}

//This is the constructor for denseLayer, we just have to initialise the layer 
//With a bunch of weights and biases, we have many methods for this one
// 1) all 0s -> creates symmetry problem, all neurons will learn the same thing
// 2) all 1s -> also creates symmetry problem, all neurons will learn the same thing
// 3) random values -> this is the best option, it breaks the symmetry -> use this
DenseLayer::DenseLayer(int in_size, int out_size, std::shared_ptr<Optimizer> opt, std::string name, act_type _act) {
    //wts dims -> outputDim * inputDim
    //input to frwd func -> inputDim * batchSize
    //bias dims -> outputDim * batchSize

    float scale = std::sqrt(6.0f / in_size); // He Initialization
    weights = Eigen::MatrixXf::Random(out_size, in_size) * scale;
    biases = Eigen::VectorXf::Zero(out_size); // Start biases at 0
    activator = Activator(_act); // store the activation type for this layer
    optimizer = opt;
    layer_name = name;

} 

// This is the Forward function for the denseLayers
Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    // We also need to save the last Inputs for backpropogation
    input_cache = input; 
    
    // Perform linear transformation: WX + B
    z_cache = computeLinearOp(input); 

    // Apply the Activation function
    return activator.applyActivation(z_cache);
}

// Helper to handle the raw matrix math WX + B
Eigen::MatrixXf DenseLayer::computeLinearOp(const Eigen::MatrixXf& input) {
    // W(o * i) * X(i * BS) + B(o * BS) = Output(o * BS)
    return (weights * input).colwise() + biases;
}

//we use 2 types of layers, ReLU layer and Softmax Layer at the output
//None Activation is just a placeholder linear layer, never to be reached

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {
    // 1. Calculate the activation gradient (Delta)
    Eigen::MatrixXf delta = activator.applyActivationGrad(grad_output, z_cache); 
    
    // 2. Calculate gradients for the weights and biases
    computeParameterGrads(delta);

    // std::cout << "Delta norm: " << delta.norm() << std::endl; // Debugging line to check if delta is not zero

    // 3. Return the gradient for the previous layer (dL/dInput = W^T * delta)
    return weights.transpose() * delta; 
}


// Helper to calculate dL/dW and dL/dB
void DenseLayer::computeParameterGrads(const Eigen::MatrixXf& delta) {
    float inv_batch = 1.0f/static_cast<float>(input_cache.cols()); // to average the gradients over the batch
    weight_grads = (delta * input_cache.transpose()) * inv_batch; // dL/dW = delta * input^T
    bias_grads = delta.rowwise().sum() * inv_batch; // dL/dB = sum of delta across the batch
}

void DenseLayer::update() {
    // Update weights and biases using the optimizer
    optimizer->update(weights, weight_grads, layer_name + "_w");
    optimizer->update(biases, bias_grads, layer_name + "_b");
} 







//TESTING AND SAVING/LOADING FUNCTIONS
void DenseLayer::setParameters(const Eigen::MatrixXf& w, const Eigen::VectorXf& b) {
    weights = w;
    biases = b;
} 

void DenseLayer::setGrads(const Eigen::MatrixXf& wg, const Eigen::VectorXf& bg) {
    weight_grads = wg;
    bias_grads = bg;
}

Eigen::MatrixXf DenseLayer::getWeights() const { return weights; }
Eigen::VectorXf DenseLayer::getBiases() const { return biases; }
Eigen::MatrixXf DenseLayer::getWeightGrads() const { return weight_grads; }
Eigen::VectorXf DenseLayer::getBiasGrads() const { return bias_grads; }