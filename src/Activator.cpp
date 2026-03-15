#include <Activator.hpp>

Activator::Activator(act_type _act) : activationType(_act) {};

Eigen::MatrixXf Activator::applyActivation(const Eigen::MatrixXf& z) const {
    if(activationType == act_type::None) {
        return z; // just return the linear output
    } else if(activationType == act_type::ReLU) {
        return z.unaryExpr([](float val) {
            return val > 0 ? val : 0.0f; // ReLU activation function
        });
    } else if(activationType == act_type::Softmax) {
        // Inside Activator::applyActivation for Softmax
        Eigen::RowVectorXf maxes = z.colwise().maxCoeff();
        Eigen::MatrixXf expZ = (z.rowwise() - maxes).array().exp();
        Eigen::RowVectorXf sumExpZ = expZ.colwise().sum();

        // 1. Divide by sum (adding epsilon to prevent division by zero)
        Eigen::MatrixXf soft = expZ.array().rowwise() / sumExpZ.array();

        // 2. Manual Clip: Ensure no value is exactly 0.0 or 1.0
        // We use cwiseMax to set a floor and cwiseMin to set a ceiling
        return soft.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
    } else if(activationType == act_type::LeReLU) {
        return z.unaryExpr([](float val) {
            return val > 0 ? val : 0.01f * val; // Leaky ReLU activation function
        });
    }
    return z; // default to linear if unknown type
}


Eigen::MatrixXf Activator::applyActivationGrad(const Eigen::MatrixXf& grad_out, const Eigen::MatrixXf& z) const {
    if(activationType == act_type::ReLU) {
        return grad_out.array() * z.unaryExpr([](float zi) {
            return zi > 0 ? 1.0f : 0.0f; // Derivative of ReLU
        }).array(); 
    } else if (activationType == act_type::Softmax || activationType == act_type::None) {
        // For Softmax, we assume the loss function handles the (y_pred - y_true) simplification
        return grad_out; 
    } else if(activationType == act_type::LeReLU) {
        return grad_out.array() * z.unaryExpr([](float zi) {
            return zi > 0 ? 1.0f : 0.01f; // Derivative of Leaky ReLU
        }).array(); 
    }
}