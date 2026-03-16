#include <iostream>
#include <Network.hpp>
#include <Loss.hpp>
#include <random>

void train(NeuralNetwork& net, Eigen::MatrixXf& X, Eigen::MatrixXf& Y, int epochs, int batch_size) {
    net.setTraining(true); // Regularization ON
    CrossEntropyLoss loss_func;
    
    std::vector<int> indices(X.cols());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), g); 
        float total_loss = 0;

        for (int i = 0; i < X.cols(); i += batch_size) {
            int current_batch = std::min(batch_size, (int)X.cols() - i);
            Eigen::MatrixXf x_batch(X.rows(), current_batch);
            Eigen::MatrixXf y_batch(Y.rows(), current_batch);

            for(int b = 0; b < current_batch; ++b) {
                x_batch.col(b) = X.col(indices[i + b]);
                y_batch.col(b) = Y.col(indices[i + b]);
            }

            Eigen::MatrixXf output = net.forward(x_batch);
            total_loss += loss_func.calculate(output, y_batch);
            net.backward(output - y_batch);
            net.update();
        }
        std::cout << "Epoch " << epoch + 1 << " | Avg Loss: " << total_loss / (X.cols() / batch_size) << std::endl;
    }
}