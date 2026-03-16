#include <Regulator.hpp>

BatchnormRegulator::BatchnormRegulator(int features, float momentum, float eps)
    : num_features(features), momentum(momentum), epsilon(eps), is_training(true) {
    gamma = Eigen::VectorXf::Ones(features);
    beta = Eigen::VectorXf::Zero(features);
    running_mean = Eigen::VectorXf::Zero(features);
    running_var = Eigen::VectorXf::Ones(features);
}

void BatchnormRegulator::setTrainingMode(bool training) { is_training = training; }

void BatchnormRegulator::forward(Eigen::MatrixXf& activations) {
    if (is_training) {
        Eigen::VectorXf batch_mean = activations.rowwise().mean();
        Eigen::VectorXf batch_var = (activations.colwise() - batch_mean).array().square().rowwise().mean();
        
        batch_std_inv = (batch_var.array() + epsilon).sqrt().inverse();
        x_hat = (activations.colwise() - batch_mean).array().colwise() * batch_std_inv.array();
        
        activations = (x_hat.array().colwise() * gamma.array()).colwise() + beta.array();
        
        running_mean = momentum * running_mean + (1.0f - momentum) * batch_mean;
        running_var = momentum * running_var + (1.0f - momentum) * batch_var;
    } else {
        Eigen::VectorXf std_inv = (running_var.array() + epsilon).sqrt().inverse();
        activations = (((activations.colwise() - running_mean).array().colwise() * std_inv.array()).colwise() * gamma.array()).colwise() + beta.array();
    }
}

void BatchnormRegulator::backward(Eigen::MatrixXf& gradients) {
    int m = static_cast<int>(gradients.cols());
    
    d_gamma = (gradients.array() * x_hat.array()).rowwise().sum();
    d_beta = gradients.rowwise().sum();
    
    Eigen::MatrixXf dx_hat = gradients.array().colwise() * gamma.array();
    
    // Batchnorm Backprop formula
    gradients = (1.0f / m) * batch_std_inv.asDiagonal() * (
        m * dx_hat.array() - 
        dx_hat.rowwise().sum().array() - 
        x_hat.array().colwise() * (dx_hat.array() * x_hat.array()).rowwise().sum().array()
    ).matrix();
}

void BatchnormRegulator::update(std::shared_ptr<Optimizer> opt, const std::string& layer_name) {
    opt->update(gamma, d_gamma, layer_name + "_bn_gamma");
    opt->update(beta, d_beta, layer_name + "_bn_beta");
}
