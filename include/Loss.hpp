#ifndef LOSS_HPP
#define LOSS_HPP
#include <Eigen/Dense>

float compute_cross_entropy(const Eigen::MatrixXf& y_pred, const Eigen::MatrixXf& y_true);

// we wrap this function into a simple class
class CrossEntropyLoss {
public:
    // Calc scalar loss value
    float calculate(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets);

    // Returns the initial gradient (dL/dZ) 
    inline Eigen::MatrixXf get_gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) { return predictions - targets;}
};

#endif // LOSS_HPP