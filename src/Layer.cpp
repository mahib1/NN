#include "Layer.hpp"

Eigen::MatrixXf Layer::forward(const Eigen::MatrixXf& input) {
    return input; //Placeholder
}

//This is the constructor for denseLayer, we just have to initialise the layer 
//With a bunch of weights and biases, we have many methods for this one
// 1) all 0s -> creates symmetry problem, all neurons will learn the same thing
// 2) all 1s -> also creates symmetry problem, all neurons will learn the same thing
// 3) random values -> this is the best option, it breaks the symmetry -> use this
DenseLayer::DenseLayer(int in_size, int out_size, Activation activation) {
    //wts dims -> outputDim * inputDim
    //input to frwd func -> inputDim * batchSize
    //bias dims -> outputDim * batchSize
    weights = Eigen::MatrixXf::Random(out_size, in_size); // random initialization
    biases = Eigen::VectorXf::Random(out_size); // random initialization
    act_type = activation; // store the activation type for this layer

    weights_v = Eigen::MatrixXf::Zero(out_size, in_size); // initialize RMSProp velocity terms to zero
    biases_v = Eigen::VectorXf::Zero(out_size); // initialize RMSProp velocity terms to zero
    weights_m = Eigen::MatrixXf::Zero(out_size, in_size); // initialize momentum terms to zero
    biases_m = Eigen::VectorXf::Zero(out_size); // initialize momentum terms to zero
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
    z_cache = (weights * input).colwise() + biases; // we get the Z matrix, 
    //Now we do the Thresholding! f(zi) = 1 if zi > threshold else 0


    return Act(); // Apply the Activation function

}

Eigen::MatrixXf DenseLayer::Act() {
    if(act_type == Activation::None) {
        return z_cache; // if no activation, just return the linear output
    } else if(act_type == Activation::ReLU) {
        Eigen::MatrixXf ActMatrix = z_cache.unaryExpr([](float z) {
            return z > 0 ? z : 0.0f; // ReLU activation function
        });
        return ActMatrix;

    } else if(act_type == Activation::Softmax) {
        Eigen::RowVectorXf maxes = z_cache.colwise().maxCoeff(); // for numerical stability
        Eigen::MatrixXf expZ = (z_cache.rowwise() - maxes).array().exp(); // exponentiate the stabilized Z
        Eigen::RowVectorXf sumExpZ = expZ.colwise().sum(); // sum of exponentials for each column
        Eigen::MatrixXf ActMatrix = expZ.array().rowwise() / sumExpZ.array(); // softmax output

        return ActMatrix;
    }

    return z_cache; // default return, should never reach here
}








//Tetsing Functions
void DenseLayer::DevDenseLayer(const Eigen::MatrixXf& w, const Eigen::VectorXf& b) {
    weights = w;
    biases = b;
} // to manually initialise weights and biases for each layer for testing

