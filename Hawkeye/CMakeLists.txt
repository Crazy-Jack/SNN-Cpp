cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hawkeye)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(hawkeye ${PROJECT_SOURCE_DIR}/example/hawkeye_algorithm.cc)
set_target_properties(hawkeye PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(hawkeye ${PROJECT_SOURCE_DIR}/lib/config1.a)
set_property(TARGET hawkeye PROPERTY CXX_STANDARD 14)
set_property(TARGET hawkeye PROPERTY PROPERTY CXX_STANDARD_REQUIRED ON)


