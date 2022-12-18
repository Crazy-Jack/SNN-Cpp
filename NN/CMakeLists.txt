cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(check)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(check ${PROJECT_SOURCE_DIR}/check.cpp)
# target_link_libraries(check "${TORCH_LIBRARIES}")
# set_property(TARGET check PROPERTY CXX_STANDARD 14)
# set_property(TARGET check PROPERTY PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(simple ${PROJECT_SOURCE_DIR}/simple.cpp)
target_link_libraries(simple "${TORCH_LIBRARIES}")
set_property(TARGET simple PROPERTY CXX_STANDARD 14)
set_property(TARGET simple PROPERTY PROPERTY CXX_STANDARD_REQUIRED ON)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
