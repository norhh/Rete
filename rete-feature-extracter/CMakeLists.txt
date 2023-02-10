cmake_minimum_required(VERSION 3.4.3)
project(rete)

set(LLVM_DIR "${RETE_LLVM}/share/llvm/cmake/")
set(Clang_DIR "${RETE_LLVM}/share/clang/cmake/")
set(F1X_CLANG_INCLUDE "${RETE_LLVM}/lib/clang/3.8.1/include/")

# LLVM
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})


# Clang
find_package(Clang REQUIRED CONFIG)



# Compiler
set(CMAKE_CXX_STANDARD 11) # C++11...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

# The version number.
set (RETE_VERSION_MAJOR 0)
set (RETE_VERSION_MINOR 1)
set (RETE_VERSION_PATCH 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/Config.h.in"
  "${PROJECT_BINARY_DIR}/Config.h"
  )

configure_file (
"${PROJECT_SOURCE_DIR}/Globals.h.in"
"${PROJECT_BINARY_DIR}/Globals.h"
)

# add the binary tree to the search path for include files
# so that we will find Config.h
include_directories("${PROJECT_BINARY_DIR}")

include_directories(rete)
include_directories(thirdparty)

add_subdirectory(rete)
add_subdirectory(tools)

set (RETE_TOOLS_DIR "${PROJECT_BINARY_DIR}/tools/")

