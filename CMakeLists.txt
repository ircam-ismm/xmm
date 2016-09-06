cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

# Project declaration
project(xmm)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# The version number.
set (xmm_VERSION_MAJOR 0)
set (xmm_VERSION_MINOR 1)

# Compiler Flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -DJSON_DEBUG -fPIC")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Include JSON
include_directories(./dependencies/jsoncpp/include)
include_directories(./dependencies/jsoncpp/src)

# Declare library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core/distributions)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core/trainingset)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models/gmm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models/hmm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models/kmeans)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
add_definitions(-DUSE_PTHREAD)

file(
    GLOB_RECURSE
    xmm_source_files
    src/*
)

file(
    GLOB_RECURSE
    jsoncpp_source_files
    dependencies/jsoncpp/src/*
)

add_library(
    xmm
    STATIC
    ${xmm_source_files} ${jsoncpp_source_files}
)

# Declare Unit tests
set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/catch)

# message(${LINK_DIRECTORIES})

file(
    GLOB_RECURSE
    xmm_test_files
    test/*
)

add_executable(
    tests
    EXCLUDE_FROM_ALL
    ${xmm_test_files}
)

# linking configuration
target_link_libraries(
    tests
    xmm
)

set_target_properties(tests PROPERTIES OUTPUT_NAME xmm_testing)
add_custom_command(TARGET tests POST_BUILD COMMAND ${EXECUTABLE_OUTPUT_PATH}/xmm_testing)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)

find_program(DOTPATH dot)
if(${DOTPATH} STREQUAL "DOTPATH-NOTFOUND")
  set(HAS_DOT "")
else(${DOTPATH} STREQUAL "DOTPATH-NOTFOUND")
  set(HAS_DOT "-dot")
endif(${DOTPATH} STREQUAL "DOTPATH-NOTFOUND")

if(DOXYGEN_FOUND)
    add_custom_target(doc
                      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile${HAS_DOT}
                      DEPENDS ${tracking_source_files}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/
                      COMMENT "Generating API documentation with Doxygen" VERBATIM)
else(DOXYGEN_FOUND)
    message(STATUS "Doxygen not found, doc will not be included in the Makefile")
endif(DOXYGEN_FOUND)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/doc/html")
