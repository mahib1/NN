#include "Layer.hpp"

Eigen::MatrixXf Layer::forward(const Eigen::MatrixXf& input) {
    return input; //Placeholder
}

//This is the constructor for denseLayer, we just have to initialise the layer 
//With a bunch of weights and biases, we have many methods for this one
// 1) all 0s -> creates symmetry problem, all neurons will learn the same thing
// 2) all 1s -> also creates symmetry problem, all neurons will learn the same thing
// 3) random values -> this is the best option, it breaks the symmetry -> use this
DenseLayer::DenseLayer(int in_size, int out_size) {
    //wts dims -> outputDim * inputDim
    //input to frwd func -> inputDim * batchSize
    //bias dims -> outputDim * batchSize
    weights = Eigen::MatrixXf::Random(out_size, in_size); // random initialization
    biases = Eigen::VectorXf::Random(out_size); // random initialization
} // NOTE: we will discard this random function for golrot/Kaiming initialisation manually
//but for now, this is good enough imo

// This is the Forward function for the denseLayers
Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    // we need to do WX + B here , 
    // W dimension is output * input dims 
    // X dim is Input * batchSize 
    // B dim is output * batchSize
    // NOTE: W(o * i) * X(i * BS) + B(o * BS) = Output(o * BS) -> input to next layer

    // We also need to save the last Inputs for backpropogation
    input_cache = input; 
    Eigen::MatrixXf Z = (weights * input).colwise() + biases; // we get the Z matrix, 
    //Now we do the Thresholding! f(zi) = 1 if zi > threshold else 0

    Eigen::MatrixXf Act = Activation(Z, [](float z) {
        return z > 0 ? 1.0f : 0.0f; // ReLU activation function
    }); // dimension of this Act matrix is just outputDim * batchSize

    return Act;
}

Eigen::MatrixXf DenseLayer::Activation(const Eigen::MatrixXf& z, std::function<float(float)> activation_func) {
    return z.unaryExpr(activation_func);
}








//Tetsing Functions
void DenseLayer::DevDenseLayer(const Eigen::MatrixXf& w, const Eigen::VectorXf& b) {
    weights = w;
    biases = b;
} // to manually initialise weights and biases for each layer for testing

