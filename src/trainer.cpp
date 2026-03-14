#include <iostream>
#include <Network.hpp>
#include <Loss.hpp>

void train(NeuralNetwork& net, const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, int epochs, int batch_size) {
    CrossEntropyLoss loss_func;
    float learning_rate = 0.001f; // Standard for Adam

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0;
        
        // Loop through data in mini-batches
        for (int i = 0; i < X.cols(); i += batch_size) {
            int current_batch = std::min(batch_size, (int)X.cols() - i);
            
            // Extract batch using Eigen's .block()
            Eigen::MatrixXf x_batch = X.block(0, i, X.rows(), current_batch);
            Eigen::MatrixXf y_batch = Y.block(0, i, Y.rows(), current_batch);

            // 1. Forward pass
            Eigen::MatrixXf output = net.forward(x_batch);
            
            // 2. Compute Loss
            total_loss += loss_func.calculate(output, y_batch);

            // 3. Backward pass (using the Softmax-CE shortcut: grad = pred - true)
            Eigen::MatrixXf grad = output - y_batch;
            net.backward(grad);

            // 4. Update Weights
            net.update(learning_rate);
        }
        
        std::cout << "Epoch " << epoch + 1 << " | Avg Loss: " << total_loss / (X.cols() / batch_size) << std::endl;
    }
}