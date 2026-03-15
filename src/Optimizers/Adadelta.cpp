#include <Optimizer.hpp>


//
//class AdadeltaOptimizer : public Optimizer {
//private:
//    float decay, ep; 
//
//    //run avg of sq grads
//    std::map<std::string, Eigen::MatrixXf> E_g2_m; 
//    std::map<std::string, Eigen::VectorXf> E_g2_b;
//
//    //run avg of squared updates (deltas)
//    std::map<std::string, Eigen::MatrixXf> E_delta2_m;
//    std::map<std::string, Eigen::VectorXf> E_delta2_b;
//
//public:
//    AdadeltaOptimizer(float decay = 0.95f, float ep = 1e-8f) : decay(decay), ep(ep) {};
//    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
//    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override;
//};


void AdadeltaOptimizer::update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) {
    if(E_g2_m.find(id) == E_g2_m.end()) {
        E_g2_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
        E_delta2_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
    }    

    // Update running average of squared gradients
    E_g2_m[id] = decay * E_g2_m[id] + (1.0f - decay) * grads.array().square().matrix();

    // Compute update (delta)
    Eigen::MatrixXf delta = (grads.array() * (E_delta2_m[id].array() + ep).sqrt() / (E_g2_m[id].array() + ep).sqrt()).matrix();

    // Update parameters
    params -= delta;

    // Update running average of squared updates
    E_delta2_m[id] = decay * E_delta2_m[id] + (1.0f - decay) * delta.array().square().matrix();
}

void AdadeltaOptimizer::update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) {
    if(E_g2_b.find(id) == E_g2_b.end()) {
        E_g2_b[id] = Eigen::VectorXf::Zero(params.rows());
        E_delta2_b[id] = Eigen::VectorXf::Zero(params.rows());
    }    

    // Update running average of squared gradients
    E_g2_b[id] = decay * E_g2_b[id] + (1.0f - decay) * grads.array().square().matrix();

    // Compute update (delta)
    Eigen::VectorXf delta = (grads.array() * (E_delta2_b[id].array() + ep).sqrt() / (E_g2_b[id].array() + ep).sqrt()).matrix();

    // Update parameters
    params -= delta;

    // Update running average of squared updates
    E_delta2_b[id] = decay * E_delta2_b[id] + (1.0f - decay) * delta.array().square().matrix();
}