#include <Network.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

Eigen::MatrixXf NeuralNetwork::forward(Eigen::MatrixXf input) {
    for (auto& layer : layers) {
        input = layer -> forward(input);
    }
    return input;
}

void NeuralNetwork::backward(Eigen::MatrixXf initial_grad) {
    Eigen::MatrixXf grad = initial_grad;
    for (int i = layers.size() - 1; i >= 0; i--) {
        grad = layers[i] -> backward(grad);
    }
}

void NeuralNetwork::update() {
    for (auto& layer : layers) {
        layer -> update();
        // std::cout << layer -> getWeights().norm() << std::endl; // Debug: print weight norm after update
    }

    if(!layers.empty()) {
        layers[0] -> getOptimizer() -> step();
    }
}

void NeuralNetwork::save(const std::string& folder_name) {
    // Create the directory if it doesn't exist
    std::filesystem::create_directory(folder_name);

    for (int i = 0; i < layers.size(); ++i) {
        if(!(layers[i] -> hasParameters())) continue; 
        std::string w_path = folder_name + "/layer_" + std::to_string(i) + "_w.bin";
        std::string b_path = folder_name + "/layer_" + std::to_string(i) + "_b.bin";

        std::ofstream w_file(w_path, std::ios::binary);
        std::ofstream b_file(b_path, std::ios::binary);

        Eigen::MatrixXf w = layers[i] -> getWeights();
        Eigen::VectorXf b = layers[i] -> getBiases();

        // Write raw memory buffer to disk
        w_file.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
        b_file.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(float));
    }
    std::cout << "Model weights saved to " << folder_name << std::endl;
}

void NeuralNetwork::load(const std::string& folder_name) {
    for (int i = 0; i < layers.size(); ++i) {
        if(!(layers[i] -> hasParameters())) continue;
        std::string w_path = folder_name + "/layer_" + std::to_string(i) + "_w.bin";
        std::string b_path = folder_name + "/layer_" + std::to_string(i) + "_b.bin";

        std::ifstream w_file(w_path, std::ios::binary);
        std::ifstream b_file(b_path, std::ios::binary);

        if (!w_file || !b_file) {
            throw std::runtime_error("Could not find weights in " + folder_name);
        }

        // Create temporary matrices to read into
        Eigen::MatrixXf w = layers[i] -> getWeights();
        Eigen::VectorXf b = layers[i] -> getBiases();

        w_file.read(reinterpret_cast<char*>(w.data()), w.size() * sizeof(float));
        b_file.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(float));

        // Use your existing DevDenseLayer or a new setter to apply them
        layers[i] -> setParameters(w, b);
    }
    std::cout << "Model weights loaded successfully!" << std::endl;
}