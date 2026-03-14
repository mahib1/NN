#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

class Layer {
public: 
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    virtual ~Layer() = default;
}; // dummy virtual class to implement layer functions

class DenseLayer : public Layer { // dense layer extends the layer class
private:
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;
    Eigen::MatrixXf input_cache; // last input cached for backpropogation

public:
    DenseLayer(int in_size, int out_size, int batchSize);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

private:
    // Dev Testing functions to check if things work correctly!
    //we need a function to initialise weights and biases manually
    void DevDenseLayer(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b);
};

#endif