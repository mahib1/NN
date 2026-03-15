#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <Eigen/Dense>

class Optimizer {
public:
    virtual ~Optimizer() = default; 

    virtual void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& param_id) = 0; // for weights
    virtual void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& param_id) = 0; // for biases
    virtual void step() {}; 
};

//done
class SGDOptimizer : public Optimizer {
private:
    float lr; //Learning rate

public:
    SGDOptimizer(float lr) : lr(lr) {}; 
    inline void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override { params -= lr * grads; };
    inline void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override { params -= lr * grads; };
};


//done
class AdamOptimizer : public Optimizer {
private:
    float lr, B1, B2, ep; 
    int t = 1; 
    std::map<std::string, Eigen::MatrixXf> m_m, v_m; //map to store momentum, velocity for wts
    std::map<std::string, Eigen::VectorXf> m_b, v_v; //map to store momentum, velo for biases

public:
    AdamOptimizer(float lr = 0.001f, float B1 = 0.9f, float B2 = 0.999f, float ep = 1e-8f) : lr(lr), B1(B1), B2(B2), ep(ep) {};
    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override; 
    void step() override { t++; }
};


//done
class RMSPropOptimizer : public Optimizer {
private:
    float lr, decay, ep; 
    std::map<std::string, Eigen::MatrixXf> v_m; 
    std::map<std::string, Eigen::VectorXf> v_b; 

public:
    RMSPropOptimizer(float lr = 0.001f, float decay = 0.9f, float ep = 1e-8f) : lr(lr), decay(decay), ep(ep) {}; 
    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override; 
};  


//done
class AdagradOptimizer : public Optimizer {
private:
    float lr, ep; 
    std::map<std::string, Eigen::MatrixXf> v_m; //map to store cache for wts
    std::map<std::string, Eigen::VectorXf> v_b; //map to store cache for biases

public:
    AdagradOptimizer(float lr = 0.001f, float ep = 1e-8f) : lr(lr), ep(ep) {}; 
    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override; 

};

//done 
class MomentumOptimizer : public Optimizer {
private:
    float lr, momentum; 
    std::map<std::string, Eigen::MatrixXf> velocity_m; //map to store velocity for wts
    std::map<std::string, Eigen::VectorXf> velocity_b; //map to store velocity for biases   

public:
    MomentumOptimizer(float lr = 0.01f, float momentum = 0.9f) : lr(lr), momentum(momentum) {};
    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override;
};

//done
class AdadeltaOptimizer : public Optimizer {
private:
    float decay, ep; 

    //run avg of sq grads
    std::map<std::string, Eigen::MatrixXf> E_g2_m; 
    std::map<std::string, Eigen::VectorXf> E_g2_b;

    //run avg of squared updates (deltas)
    std::map<std::string, Eigen::MatrixXf> E_delta2_m;
    std::map<std::string, Eigen::VectorXf> E_delta2_b;

public:
    AdadeltaOptimizer(float decay = 0.95f, float ep = 1e-8f) : decay(decay), ep(ep) {};
    void update(Eigen::MatrixXf& params, const Eigen::MatrixXf& grads, const std::string& id) override; 
    void update(Eigen::VectorXf& params, const Eigen::VectorXf& grads, const std::string& id) override;
};

#endif