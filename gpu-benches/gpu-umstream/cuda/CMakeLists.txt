cmake_minimum_required(VERSION 3.15)

add_library(gpu-umstream-cuda OBJECT gpu-umstream.cu)
target_include_directories(gpu-umstream-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-umstream-cuda)
target_link_libraries(gpu-umstream-cuda PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-umstream-cuda $<TARGET_OBJECTS:gpu-umstream>)
    target_link_libraries(gpu-umstream-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
endif()