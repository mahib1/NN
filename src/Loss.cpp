#include <Loss.hpp>

float compute_cross_entropy(const Eigen::MatrixXf& y_pred, const Eigen::MatrixXf& y_true) {
    float epsilon = 1e-9f; // to prevent log(0)
    //Loss = -sum(y_true * log(y_pred + epsilon)) / batch_size
    Eigen::MatrixXf logs = (y_pred.array() + epsilon).log(); 
    float total_loss = -(y_true.array() * logs.array()).sum(); 

    return total_loss / static_cast<float>(y_pred.cols()); 
}

float CrossEntropyLoss::calculate(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    float epsilon = 1e-9f; // To avoid log(0)
    auto logs = (predictions.array() + epsilon).log();
    return -(targets.array() * logs).sum() / static_cast<float>(predictions.cols());
}