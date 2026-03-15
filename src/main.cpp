#include <iostream>
#include <filesystem>
#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <cmath> 
#include "Network.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "MnistLoader.cpp"

void train(NeuralNetwork& net, const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, int epochs, int batch_size);

void evaluate(NeuralNetwork& net, const Eigen::MatrixXf& testX, const Eigen::MatrixXf& testY) {
    int correct = 0;
    Eigen::MatrixXf predictions = net.forward(testX);

    for (int i = 0; i < testX.cols(); ++i) {
        int label, pred;
        testY.col(i).maxCoeff(&label);
        predictions.col(i).maxCoeff(&pred);
        if (label == pred) correct++;
    }
    std::cout << "Test Accuracy: " << (float)correct / testX.cols() * 100.0f << "%" << std::endl;
}

void debugPrint(const Eigen::MatrixXf& input) {
  std::cout << "\n--- WHAT THE NETWORK SEES ---" << std::endl;
  for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
          float val = input(i * 28 + j, 0);
          if (val > 0.5f) std::cout << "##";
          else if (val > 0.1f) std::cout << "..";
          else std::cout << "  ";
      }
      std::cout << "\n";
  }
}

int get_prediction(const Eigen::MatrixXf& output) {
    int maxIndex = 0;
    output.col(0).maxCoeff(&maxIndex);
    return maxIndex;
}

Eigen::MatrixXf captureToEigen(const sf::Image& img) {
    Eigen::MatrixXf input(784, 1);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float pixelValue = img.getPixel(x, y).r / 255.0f;
            input(y * 28 + x, 0) = pixelValue;
        }
    }
    return input;
}

int main() { 
    // --- 1. Network Initialization ---
    NeuralNetwork model;
    std::string model_folder = "../saved_model";
    auto adam = std::make_shared<AdamOptimizer>(0.0001f);
    model.addLayer(std::make_unique<DenseLayer>(784, 128,  adam, "layer_0", act_type::ReLU));
    model.addLayer(std::make_unique<DenseLayer>(128, 256, adam, "layer_1", act_type::ReLU));
    model.addLayer(std::make_unique<DenseLayer>(256, 128, adam, "layer_2", act_type::ReLU));
    model.addLayer(std::make_unique<DenseLayer>(128, 64, adam, "layer_3", act_type::ReLU));
    model.addLayer(std::make_unique<DenseLayer>(64, 10, adam, "layer_4", act_type::LeReLU));
    model.addLayer(std::make_unique<DenseLayer>(10, 10, adam, "layer_5", act_type::Softmax));

    if(std::filesystem::exists(model_folder)) {
        std::cout << "Loading existing model from " << model_folder << "..." << std::endl;
        model.load(model_folder);
    } else {
        std::cout << "No saved model found. Initializing new model..." << std::endl;
        try {
            std::cout << "Loading MNIST Training Data..." << std::endl;
            Eigen::MatrixXf trainX = MnistLoader::loadImages("C:/Users/MAHIB/code/Projects/NN/data/train-images.idx3-ubyte");
            Eigen::MatrixXf trainY = MnistLoader::loadLabels("C:/Users/MAHIB/code/Projects/NN/data/train-labels.idx1-ubyte");
            Eigen::MatrixXf testX = MnistLoader::loadImages("C:/Users/MAHIB/code/Projects/NN/data/t10k-images.idx3-ubyte");
            Eigen::MatrixXf testY = MnistLoader::loadLabels("C:/Users/MAHIB/code/Projects/NN/data/t10k-labels.idx1-ubyte");

            std::cout << "Training START!" << std::endl;
            int epochs = 4;
            int batch_size = 1;

            train(model, trainX, trainY, epochs, batch_size);
            evaluate(model, testX, testY); 
            model.save(model_folder); 
            
            std::cout << "Training Complete!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during setup/training: " << e.what() << std::endl;
            return -1;
        }
    }

    // --- 2. Create Anti-Aliased Brush Texture ---
    const int brushRes = 128; // High res for the brush texture
    sf::Texture brushTexture;
    sf::Image brushImage;
    brushImage.create(brushRes, brushRes);

    for (int y = 0; y < brushRes; ++y) {
        for (int x = 0; x < brushRes; ++x) {
            float dx = x - brushRes / 2.0f;
            float dy = y - brushRes / 2.0f;
            float distance = std::sqrt(dx * dx + dy * dy);
            float radius = brushRes / 2.0f;

            // Smooth falloff for anti-aliasing
            float alpha = 255.0f * std::max(0.0f, 1.0f - (distance / radius));
            brushImage.setPixel(x, y, sf::Color(255, 255, 255, (sf::Uint8)alpha));
        }
    }
    brushTexture.loadFromImage(brushImage);
    brushTexture.setSmooth(true); // Bilinear filtering

    // --- 3. SFML Window Setup ---
    const int scale = 20;
    sf::RenderWindow window(sf::VideoMode(28 * scale, 28 * scale), "Neural OCR - Draw a Digit");
    window.setFramerateLimit(60);

    sf::RenderTexture canvas;
    canvas.create(28, 28);
    canvas.clear(sf::Color::Black);

    std::cout << "\n--- Ready! ---" << std::endl;
    std::cout << "Draw a digit and press SPACE to predict." << std::endl;
    std::cout << "Press 'C' to clear the canvas." << std::endl;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C) {
                canvas.clear(sf::Color::Black);
                canvas.display();
            }

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space) {
                sf::Image img = canvas.getTexture().copyToImage();
                Eigen::MatrixXf input = captureToEigen(img);
                debugPrint(input);
                
                Eigen::MatrixXf output = model.forward(input);
                int result = get_prediction(output);
                
                std::cout << "Prediction: " << result << " (Confidence: " 
                          << output(result, 0) * 100 << "%)" << std::endl;
            }
        }

        // Mouse/Pen Drawing with Anti-Aliasing
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2i pos = sf::Mouse::getPosition(window);
            float tx = pos.x / (float)scale;
            float ty = pos.y / (float)scale;

            sf::Sprite brush(brushTexture);
            // Setting the origin to the center of the brush
            brush.setOrigin(brushRes / 2.0f, brushRes / 2.0f);
            
            // radius 2.0f in a 28x28 space means a diameter of 4.0f
            float brushTargetDiameter = 3.0f; 
            float brushScale = brushTargetDiameter / (float)brushRes;
            brush.setScale(brushScale, brushScale);
            brush.setPosition(tx, ty);

            // Use Alpha Blending to allow smooth edges to build up
            canvas.draw(brush);
            canvas.display();
        }

        window.clear();
        sf::Sprite sprite(canvas.getTexture());
        sprite.setScale(scale, scale);
        window.draw(sprite);
        window.display();
    }

    return 0;
}