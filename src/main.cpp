#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <Network.hpp> // Assuming your wrapper is ready

// Function to convert SFML drawing to Eigen Matrix
// MNIST expects 28x28 normalized values (0.0 to 1.0)
Eigen::MatrixXf captureToEigen(const sf::Image& img) {
    Eigen::MatrixXf input(784, 1);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            // MNIST is typically white text on black background
            // We take the Red channel as our grayscale value
            float pixelValue = img.getPixel(x, y).r / 255.0f;
            input(y * 28 + x, 0) = pixelValue;
        }
    }
    std::cout << input; // Print the captured input for verification
    return input;
}

int main() {
    // 1. Setup Window (Scales 28x28 up by 20x for visibility)
    const int scale = 20;
    sf::RenderWindow window(sf::VideoMode(28 * scale, 28 * scale), "Write Something!");
    window.setFramerateLimit(60);

    // 2. Create drawing surface (Actual 28x28 resolution)
    sf::RenderTexture canvas;
    if (!canvas.create(28, 28)) return -1;
    canvas.clear(sf::Color::Black);

    // --- Neural Network Setup Placeholder ---
    // NeuralNetwork model; 
    // model.load("my_weights.bin"); // We will implement this soon!

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            // Clear Canvas on 'C' key
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C) {
                canvas.clear(sf::Color::Black);
                canvas.display();
                std::cout << "Canvas Cleared" << std::endl;
            }
        }

        // 3. Handle Pen/Mouse Drawing
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2i localPos = sf::Mouse::getPosition(window);
            
            // Map window coordinates (560x560) back to 28x28
            float tx = localPos.x / (float)scale;
            float ty = localPos.y / (float)scale;

            // Brush setup
            sf::CircleShape brush(1.2f); // Adjust thickness for better OCR results
            brush.setFillColor(sf::Color::White);
            brush.setOrigin(1.2f, 1.2f);
            brush.setPosition(tx, ty);

            canvas.draw(brush);
            canvas.display();
        }

        // 4. Run Prediction when Space is pressed
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
            sf::Image img = canvas.getTexture().copyToImage();
            Eigen::MatrixXf input = captureToEigen(img);

            // Eigen::MatrixXf output = model.forward(input);
            // int prediction = get_prediction(output);
            
            std::cout << "Captured 28x28 matrix. Ready for Neural Net!" << std::endl;
        }

        // Render the 28x28 canvas scaled up to the 560x560 window
        window.clear();
        sf::Sprite sprite(canvas.getTexture());
        sprite.setScale(scale, scale);
        window.draw(sprite);
        window.display();
    }

    return 0;
}