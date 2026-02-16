cmake_minimum_required(VERSION 3.15)

add_library(gpu-memcpy OBJECT cuda/cudaMemcpy.cu)
target_include_directories(gpu-memcpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-memcpy)
target_link_libraries(gpu-memcpy PUBLIC baseliner::baseliner altis-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-memcpy $<TARGET_OBJECTS:gpu-memcpy>)
    target_link_libraries(gpu-memcpy PRIVATE baseliner::baseliner altis-baseliner)
endif()