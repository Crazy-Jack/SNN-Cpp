cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(myexample ${PROJECT_SOURCE_DIR}/example/lru.cc)
set_target_properties(myexample PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(myexample ${PROJECT_SOURCE_DIR}/lib/config1.a)
set_property(TARGET myexample PROPERTY CXX_STANDARD 14)
set_property(TARGET myexample PROPERTY PROPERTY CXX_STANDARD_REQUIRED ON)


