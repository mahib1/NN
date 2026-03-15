#include <Optimizer.hpp>



//class AdagradOptimizer : public Optimizer {
//private:
//    float lr, ep; 
//    std::map<std::string, Eigen::MatrixXf> v_m; //map to store cache for wts
//    std::map<std::string, Eigen::VectorXf> v_b; //map to store cache for biases
//
//public:
//    AdagradOptimizer(float lr = 0.001f, float ep = 1e-8f) : lr(lr), ep(ep) {}; 
//    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
//    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override; 
//
//};

void AdagradOptimizer::update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) {
    if(v_m.find(id) == v_m.end()) {
        v_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
    }

    v_m[id] += grads.array().square().matrix();
    params.array() -= lr * grads.array() / (v_m[id].array().sqrt() + ep);
}

void AdagradOptimizer::update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) {
    if(v_b.find(id) == v_b.end()) {
        v_b[id] = Eigen::VectorXf::Zero(params.rows());
    }

    v_b[id] += grads.array().square().matrix();
    params.array() -= lr * grads.array() / (v_b[id].array().sqrt() + ep);
}




