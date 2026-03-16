#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <Optimizer.hpp>
#include <Regulator.hpp>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <Activator.hpp>


class Layer {
public: 
    //MAIN IMPL FUNCS
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) = 0;
    virtual void update() = 0;


    //TESTING/NETWORK FUNCS
    virtual bool hasParameters() const { return false; }
    virtual bool hasGrads() const { return false; }
    virtual Eigen::MatrixXf getWeights() const { return Eigen::MatrixXf(); }
    virtual Eigen::VectorXf getBiases() const { return Eigen::VectorXf(); }
    virtual Eigen::MatrixXf getWeightGrads() const { return Eigen::MatrixXf(); }
    virtual Eigen::VectorXf getBiasGrads() const { return Eigen::VectorXf(); } 
    virtual void setParameters(const Eigen::MatrixXf& w, const Eigen::VectorXf& b) {} 
    virtual void setGrads(const Eigen::MatrixXf& wg, const Eigen::VectorXf& bg) {}
    virtual void addRegulator(std::shared_ptr<Regulator> reg) {}
    virtual void setTraining(bool training) {}
    virtual std::shared_ptr<Optimizer> getOptimizer() const { return nullptr; }
    virtual ~Layer() = default;
}; // dummy virtual class to implement layer functions


class DenseLayer : public Layer { // dense layer extends the layer class
private:
    // Listing the members we need in this class - 
    // we need the wights and biases (current, and the previos ones to find the next set for backpropogation)
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::MatrixXf input_cache; // last input cached for backpropogation
    Eigen::MatrixXf z_cache; // last Z cached for backpropogation

    Activator activator; // to store the activation type for this layer  
    RegManager regulator; //Regulator abstraction!

    // next we need the weights and biases gradients to update the weights and biases after backpropogation
    Eigen::MatrixXf weight_grads;
    Eigen::VectorXf bias_grads;

    std::shared_ptr<Optimizer> optimizer; //Optmizer abstraction!
    std::string layer_name; // id for the optmizer

    // santa's lil helpers    
    // WX + B
    Eigen::MatrixXf computeLinearOp(const Eigen::MatrixXf& input); 

    // calc dL/dW and dL/dB
    void computeParameterGrads(const Eigen::MatrixXf& delta);

public:
    //MAIN IMPLEMENTATION FUNCTIONS
    DenseLayer(int in_size, int out_size, std::shared_ptr<Optimizer> opt, std::string name, act_type _act = act_type::ReLU);
    inline void addRegulator(std::shared_ptr<Regulator> reg) override { regulator.addRegulator(reg); }
    inline void setTraining(bool training) override { regulator.setTrainingMode(training); }
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override; 
    void update() override;






    // TESTING FUNCTIONS
    // Dev testing functions to check if things work correctly 
    // we need a function to initialise weights and biases manually
    void setParameters(const Eigen::MatrixXf& w, const Eigen::VectorXf& b) override;
    
    // manual injection of gradients for testing the optimizer logic
    void setGrads(const Eigen::MatrixXf& wg, const Eigen::VectorXf& bg) override;

    // getters to verify internal states in unit tests
    Eigen::MatrixXf getWeights() const override;
    Eigen::VectorXf getBiases() const override;
    Eigen::MatrixXf getWeightGrads() const override;
    Eigen::VectorXf getBiasGrads() const override;
    std::shared_ptr<Optimizer> getOptimizer() const override { return optimizer; }
    inline bool hasParameters() const override { return true; }
    inline bool hasGrads() const override { return true; }
};

class ConvLayer : public Layer {

};

class flattenLayer : public Layer {

};

#endif