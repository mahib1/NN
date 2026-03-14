cmake_minimum_required(VERSION 3.10)
project(NeuralNetEigen)

# 1. Configuration & Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Tell CMake where to find Eigen
# Hardcoded path as per your setup
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/eigen")

# 3. Define Shared Source Files
# We list Layer.cpp here so both the main app and tests can use it
set(COMMON_SOURCES 
    src/Layer.cpp
)

# 4. Main Application Target (Release/Production)
add_executable(main_app src/main.cpp ${COMMON_SOURCES})
target_include_directories(main_app PRIVATE 
    include 
    ${EIGEN3_INCLUDE_DIR}
)

# 5. Unit Test Target
add_executable(run_tests test/test_layer.cpp ${COMMON_SOURCES})
target_include_directories(run_tests PRIVATE 
    include 
    ${EIGEN3_INCLUDE_DIR}
)