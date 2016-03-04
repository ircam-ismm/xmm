cmake_minimum_required (VERSION 2.6)

# Project declaration
project(xmm)

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

add_subdirectory(dependencies/libjson)

# Include JSON
include_directories(./dependencies/libjson/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libjson/bin/${CMAKE_BUILD_TYPE})

# Declare library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
add_definitions(-DUSE_PTHREAD)

file(
    GLOB_RECURSE
    xmm_source_files
    src/*
)

add_library(
    xmm
    STATIC
    ${xmm_source_files}
)

# linking configuration
target_link_libraries(
    xmm
    json pthread
)

# Declare Unit tests
set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/catch)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libjson/bin/${CMAKE_BUILD_TYPE} ${LIBRARY_OUTPUT_PATH})

# message(${LINK_DIRECTORIES})

file(
    GLOB_RECURSE
    xmm_test_files
    test/*
)

add_executable(
    test
    EXCLUDE_FROM_ALL
    ${xmm_test_files}
)

# linking configuration
target_link_libraries(
    test
    xmm
)

set_target_properties(test PROPERTIES OUTPUT_NAME xmm_testing)
add_custom_command(TARGET test POST_BUILD COMMAND ${EXECUTABLE_OUTPUT_PATH}/xmm_testing)

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
    add_custom_target(docapi
                      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile-API
                      DEPENDS ${tracking_source_files}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/
                      COMMENT "Generating API documentation with Doxygen" VERBATIM)
else(DOXYGEN_FOUND)
    message(STATUS "Doxygen not found, doc will not be included in the Makefile")
endif(DOXYGEN_FOUND)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/doc/html")
