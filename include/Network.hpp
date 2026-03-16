#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Layer.hpp>
#include <vector>
#include <memory>

class NeuralNetwork {
private:
    //Layer* helps store any child class
    std::vector<std::unique_ptr<Layer>> layers; 

public:
    inline void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Eigen::MatrixXf forward(Eigen::MatrixXf input);
    void backward(Eigen::MatrixXf initial_grad); 
    
    // don't need to pass lr here if the Optimizer object 
    // already knows it. We just tell the layers to update.
    void update(); 

    inline void setTraining(bool training) { for (auto& layer : layers) layer -> setTraining(training); }

    void save(const std::string& folder_name);
    void load(const std::string& folder_name);
};

#endif