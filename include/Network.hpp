#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Layer.hpp>

class NeuralNetwork {
private:
    std::vector<DenseLayer*> layers; 

public:
    inline void addLayer(DenseLayer* layer) {layers.push_back(layer);}

    Eigen::MatrixXf forward(Eigen::MatrixXf input);
    void backward(Eigen::MatrixXf initial_grad); 
    void update(float lr); 
};


#endif // NETWORK_HPP