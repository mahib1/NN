#include <Network.hpp>

Eigen::MatrixXf NeuralNetwork::forward(Eigen::MatrixXf input) {
    for (auto& layer : layers) {
        input = layer->forward(input);
    }
    return input;
}

void NeuralNetwork::backward(Eigen::MatrixXf initial_grad) {
    Eigen::MatrixXf grad = initial_grad;
    for (int i = layers.size() - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad);
    }
}

void NeuralNetwork::update(float lr) {
    for (auto& layer : layers) {
        layer->update(lr);
    }
}