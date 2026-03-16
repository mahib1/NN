#include <Regulator.hpp>

void RegManager::addRegulator(std::shared_ptr<Regulator> reg) { regulators.push_back(reg); }

void RegManager::setTrainingMode(bool training) {
    for (auto& reg : regulators) reg->setTrainingMode(training);
}

void RegManager::applyForward(Eigen::MatrixXf& activations) {
    for (auto& reg : regulators) reg->forward(activations);
}

void RegManager::applyBackward(Eigen::MatrixXf& gradients) {
    for (int i = static_cast<int>(regulators.size()) - 1; i >= 0; --i) {
        regulators[i]->backward(gradients);
    }
}

void RegManager::applyUpdate(std::shared_ptr<Optimizer> opt, const std::string& layer_name) {
    for (auto& reg : regulators) {
        if (reg->hasParameters()) {
            reg->update(opt, layer_name);
        }
    }
}