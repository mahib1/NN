cmake_minimum_required(VERSION 3.14) # Minimum for FetchContent
project(NeuralNetEigen)

# 1. Configuration & Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Parallel Computing (OpenMP)
find_package(OpenMP)

# 3. SFML Dependency Management (FetchContent)
include(FetchContent)

# We try to find SFML on the system first to save build time
find_package(SFML 2.6 COMPONENTS graphics window system QUIET)

if(NOT SFML_FOUND)
    message(STATUS "SFML not found locally. Downloading and building from source...")
    
    # Configure SFML to build STATICALLY 
    # This means no .dll or .so files needed!
    set(SFML_STATIC_LIBRARIES ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    
    # Disable components you don't use to speed up build
    set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
    set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)
    set(SFML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SFML_BUILD_DOC OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        sfml
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG        2.6.1
    )
    FetchContent_MakeAvailable(sfml)
endif()

# 4. Define Source Files
# Using GLOB_RECURSE to ensure all optimizer implementations are caught
file(GLOB_RECURSE OPTIMIZER_SOURCES "src/Optimizers/*.cpp")

set(COMMON_SOURCES 
    src/Layer.cpp
    src/Loss.cpp
    src/Network.cpp
    src/Activator.cpp
    src/trainer.cpp
    ${OPTIMIZER_SOURCES}
)

# 5. Main Application Target
add_executable(main_app src/main.cpp ${COMMON_SOURCES})

# Include directories
target_include_directories(main_app PRIVATE 
    include 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/eigen"
)

# Link Libraries
# Note: Using PRIVATE keywords for modern CMake consistency
target_link_libraries(main_app PRIVATE 
    sfml-graphics 
    sfml-window 
    sfml-system
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main_app PRIVATE OpenMP::OpenMP_CXX)
endif()

# 6. Compiler-Specific Optimizations
if(MSVC)
    # /W4: high warning level, /Ox: max optimization, /arch:AVX2: SIMD
    target_compile_options(main_app PRIVATE /W4 /Ox /arch:AVX2)
    # Ensure static linking for MSVC runtime as well if desired
    set_target_properties(main_app PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # -O3: max optimization, -march=native: optimize for current CPU
    target_compile_options(main_app PRIVATE -Wall -O3 -march=native)
endif()

# 7. Unit Test Target
add_executable(run_tests test/test_dense_layer.cpp ${COMMON_SOURCES})
target_include_directories(run_tests PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/include/eigen")
target_link_libraries(run_tests PRIVATE) # Initialize

if(OpenMP_CXX_FOUND)
    target_link_libraries(run_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(main_app PRIVATE 
  PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
