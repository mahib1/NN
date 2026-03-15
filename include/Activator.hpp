#ifndef ACTIVATOR_HPP
#define ACTIVATOR_HPP

#include <Eigen/Dense>
enum act_type {
    ReLU,
    Softmax,
    LeReLU,
    None
};  

class Activator {
public:
    act_type activationType;

    Activator(act_type _act = act_type::ReLU);
    Eigen::MatrixXf applyActivation(const Eigen::MatrixXf& z) const;
    Eigen::MatrixXf applyActivationGrad(const Eigen::MatrixXf& grad_out, const Eigen::MatrixXf& z) const;
};


#endif