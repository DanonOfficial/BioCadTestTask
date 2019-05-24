cmake_minimum_required(VERSION 3.13)
project(BioCadTestTask)

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
include_directories( ${OPENCL_INCLUDE_DIR} )

add_executable(BioCadTestTask main.cpp)



target_link_libraries(BioCadTestTask OpenCL::OpenCL)