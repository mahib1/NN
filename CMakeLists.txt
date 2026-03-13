cmake_minimum_required(VERSION 3.10)
project(EigenTest)

# 1. Tell CMake where to find Eigen
# If installed system-wide, use: find_package(Eigen3 REQUIRED)
# If you just downloaded the folder, set the path:
set(EIGEN3_INCLUDE_DIR "C:\\Users\\MAHIB\\code\\git\\eigen")

add_executable(EigenTest src/main.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Add the include directory to your target
target_include_directories(EigenTest PRIVATE ${EIGEN3_INCLUDE_DIR})
