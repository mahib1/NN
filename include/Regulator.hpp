#ifndef REGULATOR_HPP
#define REGULATOR_HPP

#include <Eigen/Dense>
#include <Optimizer.hpp>
#include <vector>
#include <memory>
#include <string>

class Regulator {
public:
    virtual ~Regulator() = default;
    virtual void setTrainingMode(bool training) = 0;
    virtual void forward(Eigen::MatrixXf& activations) = 0;
    virtual void backward(Eigen::MatrixXf& gradients) = 0;

    virtual bool hasParameters() const { return false; }
    virtual void update(std::shared_ptr<Optimizer> opt, const std::string& layer_name) {}
};

class DropoutRegulator : public Regulator {
private:
    float keep_prob;
    bool is_training;
    Eigen::MatrixXf mask;

public:
    DropoutRegulator(float drop_prob);
    void setTrainingMode(bool training) override;
    void forward(Eigen::MatrixXf& activations) override;
    void backward(Eigen::MatrixXf& gradients) override;
};

class BatchnormRegulator : public Regulator {
private:
    int num_features;
    float momentum;
    float epsilon;
    bool is_training;

    // trainable 
    Eigen::VectorXf gamma; 
    Eigen::VectorXf beta;  
    
    // gradients cached
    Eigen::VectorXf d_gamma;
    Eigen::VectorXf d_beta;

    // Running stats
    Eigen::VectorXf running_mean;
    Eigen::VectorXf running_var;

    // Caches for backPass
    Eigen::MatrixXf x_hat;
    Eigen::VectorXf batch_std_inv;

public:
    BatchnormRegulator(int features, float momentum = 0.9f, float eps = 1e-5f);
    
    void setTrainingMode(bool training) override;
    void forward(Eigen::MatrixXf& activations) override;
    void backward(Eigen::MatrixXf& gradients) override;
    
    //overrides to link with Optimizer
    bool hasParameters() const override { return true; }
    void update(std::shared_ptr<Optimizer> opt, const std::string& layer_name) override;
};

class RegManager {
private:
    std::vector<std::shared_ptr<Regulator>> regulators;

public:
    RegManager() = default;
    void addRegulator(std::shared_ptr<Regulator> reg);
    void setTrainingMode(bool training);
    void applyForward(Eigen::MatrixXf& activations);
    void applyBackward(Eigen::MatrixXf& gradients);
    
    // tells all regulators to talk to the optimizer
    void applyUpdate(std::shared_ptr<Optimizer> opt, const std::string& layer_name);
};

#endif