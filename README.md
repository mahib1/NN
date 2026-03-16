
# C++ Eigen-NN Framework

A high-performance, modular Neural Network library written from scratch in C++. This framework leverages **Eigen3** for vectorized linear algebra and **SFML** for interactive visualization. It supports a variety of modern deep learning features including multiple optimizers, activation functions, and regularization techniques like Batch Normalization and Dropout.

## Core Features

* **Modular Layers:** Easily stack `Dense` layers (with planned support for `Convolutional` and `Flatten` types).
* **Multiple Optimizers:** Full implementation of **Adam**, **RMSProp**, **Adagrad**, **Adadelta**, and **SGD**.
* **Advanced Regularization:** Built-in **Batch Normalization** and **Inverted Dropout** to prevent overfitting.
* **Activation Functions:** Support for **ReLU**, **Leaky ReLU**, **Softmax**, and **Linear** activations.
* **Native Performance:** Written in C++17 with Eigen3 for maximum hardware utilization and SIMD optimizations.
* **Live OCR Demo:** Interactive SFML-based GUI where you can draw digits and get real-time predictions from the MNIST-trained model.

---

## Getting Started

### Prerequisites

* **C++17 Compiler** (GCC, Clang, or MSVC)
* **CMake** (version 3.14+)
* **SFML 2.6+** (Handled via CMake FetchContent)

### Installation & Build

```bash
# Clone the repository
git clone --recursive https://github.com/mahib1/NN.git
cd NN

# Create build directory
mkdir build 
cd build 

# Configure and Build
cmake ..
cmake --build . --config Release

```

---

## Usage Example

Building a model is designed to be intuitive and modular. Below is a snippet of how to define a 4-layer network for MNIST classification:

```cpp
NeuralNetwork model;
auto adam = std::make_shared<AdamOptimizer>(0.001f);

// 784 (Input) -> 256 (ReLU)
auto l0 = std::make_unique<DenseLayer>(784, 256, adam, "l0", act_type::ReLU);
model.addLayer(std::move(l0));

// 256 -> 128 (LeReLU + BatchNorm + Dropout)
//LeReLU = Leaky ReLU
auto l1 = std::make_unique<DenseLayer>(256, 128, adam, "l1", act_type::LeReLU);
l1->addRegulator(std::make_shared<BatchnormRegulator>(128));
l1->addRegulator(std::make_shared<DropoutRegulator>(0.4f));
model.addLayer(std::move(l1));

// 128 -> 10 (Softmax Output)
model.addLayer(std::make_unique<DenseLayer>(128, 10, adam, "out", act_type::Softmax));

```

---

## Project Structure

| Directory | Description |
| --- | --- |
| `include/` | Header files for layers, optimizers, and engine. |
| `src/` | Core implementations and mathematical logic. |
| `data/` | Placeholder for MNIST `.ubyte` datasets. |
| `saved_model/` | Directory where `.bin` weights are serialized. |
| `test/` | Unit tests for mathematical verification. |

NOTE : Some Functions might be defined as inline in the header files themselves, result of my own laziness!


---

## Training vs Inference

The framework handles the internal state of regularizers automatically. Simply toggle the network state before running your passes:

* **Training Mode:** `model.setTraining(true);` (Enables Dropout masks and BatchNorm batch-statistics).
* **Inference Mode:** `model.setTraining(false);` (Disables Dropout and uses BatchNorm running averages).

---

## Future Roadmap

* [ ] **Convolutional Layers:** 2D Kernels and Pooling support.
* [ ] **Multi Core CPU and GPU Acceleration:** CUDA backend for massive matrix operations.
* [ ] **Serialization:** Support for exporting models to ONNX format.
