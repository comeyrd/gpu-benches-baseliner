cmake_minimum_required(VERSION 3.15)

add_library(gpu-umstream-hip OBJECT gpu-umstream.hip)
target_include_directories(gpu-umstream-hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-umstream-hip)
target_link_libraries(gpu-umstream-hip PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-umstream-hip $<TARGET_OBJECTS:gpu-umstream>)
    target_link_libraries(gpu-umstream-hip PRIVATE baseliner::baseliner gpu-benches-baseliner)
endif()
