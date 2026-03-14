cmake_minimum_required(VERSION 3.10)
project(NeuralNetEigen)

# 1. Configuration & Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- SFML Setup for MSVC ---
# Point this to the folder containing the 'lib' and 'include' folders of SFML
set(SFML_DIR "C:/Users/MAHIB/Downloads/SFML-2.6.1/lib/cmake/SFML")

# Find the SFML components
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)

# 2. Tell CMake where to find Eigen
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/eigen")

# 3. Define Shared Source Files
set(COMMON_SOURCES 
    src/Layer.cpp
    src/Loss.cpp
    src/Network.cpp
)

# 4. Main Application Target (Now with SFML)
add_executable(main_app src/main.cpp ${COMMON_SOURCES})
target_include_directories(main_app PRIVATE 
    include 
    ${EIGEN3_INCLUDE_DIR}
)

# Link SFML to the main application
target_link_libraries(main_app sfml-graphics sfml-window sfml-system)

# 5. Unit Test Target (Usually no SFML needed for math tests)
add_executable(run_tests test/test_dense_layer.cpp ${COMMON_SOURCES})
target_include_directories(run_tests PRIVATE 
    include 
    ${EIGEN3_INCLUDE_DIR}
)