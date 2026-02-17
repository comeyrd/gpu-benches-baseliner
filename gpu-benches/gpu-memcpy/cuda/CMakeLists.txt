cmake_minimum_required(VERSION 3.15)

add_library(gpu-memcpy-cuda OBJECT gpu-memcpy.cu)
target_include_directories(gpu-memcpy-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-memcpy-cuda)
target_link_libraries(gpu-memcpy-cuda PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-memcpy-cuda $<TARGET_OBJECTS:gpu-memcpy>)
    target_link_libraries(gpu-memcpy-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
endif()