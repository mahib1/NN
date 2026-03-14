#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

class MnistLoader {
public:
    // Flips bytes from Big-Endian (IDX) to Little-Endian (Windows/Intel)
    static uint32_t reverseInt(uint32_t i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((uint32_t)ch1 << 24) + ((uint32_t)ch2 << 16) + ((uint32_t)ch3 << 8) + ch4;
    }

    static Eigen::MatrixXf loadImages(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open image file: " + path);

        uint32_t magic = 0, num_items = 0, rows = 0, cols = 0;
        file.read((char*)&magic, 4);
        file.read((char*)&num_items, 4);
        file.read((char*)&rows, 4);
        file.read((char*)&cols, 4);

        num_items = reverseInt(num_items);
        rows = reverseInt(rows);
        cols = reverseInt(cols);

        // Matrix is (pixels_per_image x num_images)
        Eigen::MatrixXf matrix(rows * cols, num_items);
        
        for (int i = 0; i < num_items; ++i) {
            for (int j = 0; j < rows * cols; ++j) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                matrix(j, i) = (float)pixel / 255.0f; // Normalize to [0, 1]
            }
        }
        return matrix;
    }

    static Eigen::MatrixXf loadLabels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open label file: " + path);

        uint32_t magic = 0, num_items = 0;
        file.read((char*)&magic, 4);
        file.read((char*)&num_items, 4);
        num_items = reverseInt(num_items);

        // One-hot encoded matrix (10 classes x num_items)
        Eigen::MatrixXf labels = Eigen::MatrixXf::Zero(10, num_items);
        for (int i = 0; i < num_items; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, 1);
            labels((int)label, i) = 1.0f;
        }
        return labels;
    }
};
