#include <Regulator.hpp>

DropoutRegulator::DropoutRegulator(float drop_prob) 
    : keep_prob(1.0f - drop_prob), is_training(true) {}

void DropoutRegulator::setTrainingMode(bool training) { is_training = training; }

void DropoutRegulator::forward(Eigen::MatrixXf& activations) {
    if (!is_training) return;
    mask = (Eigen::MatrixXf::Random(activations.rows(), activations.cols()).array() + 1.0f) / 2.0f;
    mask = (mask.array() < keep_prob).cast<float>();
    activations = (activations.cwiseProduct(mask)) / keep_prob;
}

void DropoutRegulator::backward(Eigen::MatrixXf& gradients) {
    if (!is_training) return;
    gradients = (gradients.cwiseProduct(mask)) / keep_prob;
}
