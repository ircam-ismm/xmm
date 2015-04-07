cmake_minimum_required (VERSION 2.6)

# Project declaration
project(json)

# Declare library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/_internal/Dependencies/libbase64++/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/_internal/Dependencies/mempool++/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/_internal/Source/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/_internal/Source/JSONDefs/)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

file(
    GLOB_RECURSE
    json_source_files
    src/*
)

add_library(
    json
    STATIC
    ${json_source_files}
)