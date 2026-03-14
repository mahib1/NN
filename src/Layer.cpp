#include <Layer.hpp>
#include <cmath>

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
    
    time_step = 0; 
} 

// This is the Forward function for the denseLayers
Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    // We also need to save the last Inputs for backpropogation
    input_cache = input; 
    
    // Perform linear transformation: WX + B
    z_cache = computeLinearOp(input); 

    // Apply the Activation function
    return applyActivation(z_cache);
}

// Helper to handle the raw matrix math WX + B
Eigen::MatrixXf DenseLayer::computeLinearOp(const Eigen::MatrixXf& input) {
    // W(o * i) * X(i * BS) + B(o * BS) = Output(o * BS)
    return (weights * input).colwise() + biases;
}

Eigen::MatrixXf DenseLayer::applyActivation(const Eigen::MatrixXf& z) {
    if(act_type == Activation::None) {
        return z; // just return the linear output
    } else if(act_type == Activation::ReLU) {
        return z.unaryExpr([](float val) {
            return val > 0 ? val : 0.0f; // ReLU activation function
        });
    } else if(act_type == Activation::Softmax) {
        Eigen::RowVectorXf maxes = z.colwise().maxCoeff(); // for numerical stability
        Eigen::MatrixXf expZ = (z.rowwise() - maxes).array().exp(); // exponentiate the stabilized Z
        Eigen::RowVectorXf sumExpZ = expZ.colwise().sum(); // sum of exponentials for each column
        return expZ.array().rowwise() / sumExpZ.array(); // softmax output
    }
    return z;
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {
    // 1. Calculate the activation gradient (Delta)
    Eigen::MatrixXf delta = computeActivationGrad(grad_output); 
    
    // 2. Calculate gradients for the weights and biases
    computeParameterGrads(delta);

    // 3. Return the gradient for the previous layer (dL/dInput = W^T * delta)
    return weights.transpose() * delta; 
}

// Helper to calculate the local gradient (delta) based on the activation type
Eigen::MatrixXf DenseLayer::computeActivationGrad(const Eigen::MatrixXf& grad_output) {
    if(act_type == Activation::ReLU) {
        return grad_output.array() * z_cache.unaryExpr([](float z) {
            return z > 0 ? 1.0f : 0.0f; // derivative of ReLU
        }).array();
    } else if(act_type == Activation::Softmax || act_type == Activation::None) {
        // Note: For Softmax, we assume the loss function handles the (y_pred - y_true) simplification
        return grad_output; 
    }
    return grad_output;
}

// Helper to calculate dL/dW and dL/dB
void DenseLayer::computeParameterGrads(const Eigen::MatrixXf& delta) {
    float inv_batch = 1.0f/static_cast<float>(input_cache.cols()); // to average the gradients over the batch
    weight_grads = (delta * input_cache.transpose()) * inv_batch; // dL/dW = delta * input^T
    bias_grads = delta.rowwise().sum() * inv_batch; // dL/dB = sum of delta across the batch
}

void DenseLayer::update(float lr) {
    //Define the Hyperparameters for Adam
    time_step++;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    // Calculate the momentum and velocity for weights
    weights_m = beta1 * weights_m + (1.0f - beta1) * weight_grads; 
    weights_v = beta2 * weights_v + (1.0f - beta2) * weight_grads.array().square().matrix(); 

    biases_m = beta1 * biases_m + (1.0f - beta1) * bias_grads;
    biases_v = beta2 * biases_v + (1.0f - beta2) * bias_grads.array().square().matrix();

    // now calculate the corrected momentum and velocity for weights
    Eigen::MatrixXf m_hat = weights_m / (1.0f - std::pow(beta1, time_step)); 
    Eigen::MatrixXf v_hat = weights_v / (1.0f - std::pow(beta2, time_step)); 

    Eigen::VectorXf m_hat_b = biases_m / (1.0f - std::pow(beta1, time_step));
    Eigen::VectorXf v_hat_b = biases_v / (1.0f - std::pow(beta2, time_step));

    //finally we update the weights
    weights -= lr * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix(); 
    biases -= lr * (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix();
} 







// --- Testing & Accessor Functions ---

void DenseLayer::DevDenseLayer(const Eigen::MatrixXf& w, const Eigen::VectorXf& b) {
    weights = w;
    biases = b;
} 

void DenseLayer::setWeightGrads(const Eigen::MatrixXf& wg) { weight_grads = wg; }
void DenseLayer::setBiasGrads(const Eigen::VectorXf& bg) { bias_grads = bg; }

Eigen::MatrixXf DenseLayer::getWeights() const { return weights; }
Eigen::VectorXf DenseLayer::getBiases() const { return biases; }
Eigen::MatrixXf DenseLayer::getWeightGrads() const { return weight_grads; }
Eigen::VectorXf DenseLayer::getBiasGrads() const { return bias_grads; }