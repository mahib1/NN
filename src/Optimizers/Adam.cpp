#include <Optimizer.hpp>

//std::map<std::string, Eigen::MatrixXf> m_m, v_m; //map to store momentum, velocity for wts
//std::map<std::string, Eigen::VectorXf> m_b, m_v; //map to store momentum, velo for biases

void AdamOptimizer::update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) {
    if(m_m.find(id) == m_m.end()) {
        m_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
        v_m[id] = Eigen::MatrixXf::Zero(params.rows(), params.cols());
    }    

    m_m[id] = B1 * m_m[id] + (1 - B1) * grads;
    v_m[id] = B2 * v_m[id] + (1 - B2) * grads.array().square().matrix();

    float bc1 = 1.0f - std::pow(B1, t);
    float bc2 = 1.0f - std::pow(B2, t);

    params.array() -= lr * (m_m[id].array() / bc1) / ((v_m[id].array() / bc2).sqrt() + ep);
}

void AdamOptimizer::update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) {
    if(m_b.find(id) == m_b.end()) {
        m_b[id] = Eigen::VectorXf::Zero(params.size());
        v_v[id] = Eigen::VectorXf::Zero(params.size());
    }    

    m_b[id] = B1 * m_b[id] + (1 - B1) * grads;
    v_v[id] = B2 * v_v[id] + (1 - B2) * grads.array().square().matrix();

    float bc1 = 1.0f - std::pow(B1, t);
    float bc2 = 1.0f - std::pow(B2, t);

    params.array() -= lr * (m_b[id].array() / bc1) / ((v_v[id].array() / bc2).sqrt() + ep);

}