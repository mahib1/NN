#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

enum class Activation {
    ReLU,
    Softmax,
    None
};

class Layer {
public: 
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    virtual ~Layer() = default;
}; // dummy virtual class to implement layer functions

class DenseLayer : public Layer { // dense layer extends the layer class
private:
    // Listing the members we need in this class - 
    // we need the wights and biases (current, and the previos ones to find the next set for backpropogation)
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::MatrixXf input_cache; // last input cached for backpropogation
    Eigen::MatrixXf z_cache; // last Z cached for backpropogation

    Activation act_type; // to store the activation type for this layer 

    // next we need the weights and biases gradients to update the weights and biases after backpropogation
    Eigen::MatrixXf weight_grads;
    Eigen::VectorXf bias_grads;

    //Now we want to implement RMSProp, Adam, Momentum etc. so we need to store the previous weight and bias updates as well
    Eigen::MatrixXf weights_m; // momentum term for weights
    Eigen::VectorXf biases_m; // momentum term for biases
    Eigen::MatrixXf weights_v; // RMSProp term for weights  (velocity terms)
    Eigen::VectorXf biases_v; // RMSProp term for biases (velocity terms)

    int time_step = 0; // to keep track of the time step for Adam optimizer 

    // santa's lil helpers    
    // WX + B
    Eigen::MatrixXf computeLinearOp(const Eigen::MatrixXf& input);
    
    // activation function for the layer (handles ReLU, Softmax, None)
    Eigen::MatrixXf applyActivation(const Eigen::MatrixXf& z);
    
    // calc local gradient (delta) based on the activation type
    Eigen::MatrixXf computeActivationGrad(const Eigen::MatrixXf& grad_output);
    
    // calc dL/dW and dL/dB
    void computeParameterGrads(const Eigen::MatrixXf& delta);

public:
    DenseLayer(int in_size, int out_size, Activation activation = Activation::ReLU);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output); 
    void update(float lr);

    // Dev testing functions to check if things work correctly 
    // we need a function to initialise weights and biases manually
    void DevDenseLayer(const Eigen::MatrixXf& w, const Eigen::VectorXf& b);
    
    // manual injection of gradients for testing the optimizer logic
    void setWeightGrads(const Eigen::MatrixXf& wg);
    void setBiasGrads(const Eigen::VectorXf& bg);

    // getters to verify internal states in unit tests
    Eigen::MatrixXf getWeights() const;
    Eigen::VectorXf getBiases() const;
    Eigen::MatrixXf getWeightGrads() const;
    Eigen::VectorXf getBiasGrads() const;
};

#endif