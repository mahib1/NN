#include <Optimizer.hpp>


//class MomentumOptimizer : public Optimizer {
//private:
//    float lr, momentum; 
//    std::map<std::string, Eigen::MatrixXf> velocity_m; //map to store velocity for wts
//    std::map<std::string, Eigen::VectorXf> velocity_b; //map to store velocity for biases   
//
//public:
//    MomentumOptimizer(float lr = 0.01f, float momentum = 0.9f) : lr(lr), momentum(momentum) {};
//    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
//    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override;
//};

void MomentumOptimizer::update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) {
    if(velocity_m.find(id) == velocity_m.end()) {
        velocity_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
    }

    velocity_m[id] = 0.9f * velocity_m[id] + 0.1f * grads; // momentum update
    params -= 0.01f * velocity_m[id]; // update parameters with momentum
}   


void MomentumOptimizer::update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) {
    if(velocity_b.find(id) == velocity_b.end()) {
        velocity_b[id] = Eigen::VectorXf::Zero(params.rows());
    }

    velocity_b[id] = 0.9f * velocity_b[id] + 0.1f * grads; // momentum update
    params -= 0.01f * velocity_b[id]; // update parameters with momentum
}