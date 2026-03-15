#include <Optimizer.hpp>

//std::map<std::string, Eigen::MatrixXf> v_m; 
//std::map<std::string, Eigen::VectorXf> v_b; 
//
//public:
//RMSPropOptimizer(float lr = 0.001f, float decay = 0.9f, float ep = 1e-8f) : lr(lr), decay(decay), ep(ep) {}; 
//void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
//void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override; 

void RMSPropOptimizer::update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) {
    if(v_m.find(id) == v_m.end()) {
        v_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
    }

    v_m[id] = decay * v_m[id]  + (1.0f - decay) * grads.array().square().matrix();

    params.array() -= lr * grads.array() / (v_m[id].array().sqrt() + ep);
}  

void RMSPropOptimizer::update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) {
    if(v_b.find(id) == v_b.end()) {
        v_b[id] = Eigen::VectorXf::Zero(params.rows());
    }

    v_b[id] = decay * v_b[id]  + (1.0f - decay) * grads.array().square().matrix();

    params.array() -= lr * grads.array() / (v_b[id].array().sqrt() + ep);
}  
