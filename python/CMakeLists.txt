cmake_minimum_required (VERSION 2.6)

# Compiler Flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -DJSON_DEBUG -fPIC")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# C++11 Support
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

# Require global XMM project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/_build_tmp)

# PYTHON STUFF
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# Allow user override
if(NOT Python_EXECUTABLE)
    execute_process(
        COMMAND python -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE Python_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(Python COMPONENTS Interpreter Development)
message(STATUS "Python_LIBRARY: ${Python_LIBRARY}")
message(STATUS "Python_FOUND: ${Python_FOUND}")
message(STATUS "Python_STDLIB: ${Python_STDLIB}")
message(STATUS "Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "PYTHONLIBS_VERSION_STRING: ${PYTHONLIBS_VERSION_STRING}")
include_directories( ${Python_INCLUDE_DIRS} )

## Try to determine the directory of numpy headers
execute_process(
    COMMAND python -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
include_directories( ${NumPy_INCLUDE_DIR} )
if( NOT EXISTS ${NumPy_INCLUDE_DIR}/numpy/arrayobject.h )
    message( FATAL_ERROR "${NumPy_INCLUDE_DIR}/numpy/arrayobject.h header is not found. If it is installed, than check the python flags and numpy header directories")
endif( NOT EXISTS ${NumPy_INCLUDE_DIR}/numpy/arrayobject.h )

# Include python and XMM header directories
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/jsoncpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/jsoncpp/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/core/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/core/distributions)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/core/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/core/trainingset)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/models/gmm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/models/hmm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/models/kmeans)

# Doxygen XML documentation
find_package(Doxygen)
message(STATUS "DOXYGEN_FOUND: ${DOXYGEN_FOUND}")
if(DOXYGEN_FOUND)
    set(XML_DOC_INDEX ${CMAKE_CURRENT_SOURCE_DIR}/_build_tmp/xml/index.xml)
    add_custom_command(OUTPUT ${XML_DOC_INDEX}
                      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/Doxyfile-Python
                      DEPENDS ${tracking_source_files}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
                      COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
    add_custom_target(docpy DEPENDS ${XML_DOC_INDEX})
endif(DOXYGEN_FOUND)

# Add target for python documentation from doxygen
file(
    GLOB_RECURSE
    xmm_header_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.h
)
add_custom_command(OUTPUT xmm_doc.i
                   DEPENDS docpy ${xmm_header_files}
                   COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/src/doxy2swig.py ${CMAKE_CURRENT_SOURCE_DIR}/_build_tmp/xml/index.xml ${CMAKE_CURRENT_SOURCE_DIR}/xmm_doc.i)


set(CMAKE_SWIG_FLAGS "")
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/numpy.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/xmm.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/xmm.i PROPERTIES SWIG_FLAGS "-DUSE_PTHREAD")
SET(SWIG_MODULE_xmm_EXTRA_DEPS
    xmm_doc.i
)
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/xmm)
SWIG_ADD_LIBRARY(xmm LANGUAGE python SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/xmm.i)
SWIG_LINK_LIBRARIES(xmm ${PYTHON_LIBRARIES} xmm pthread)

add_custom_command(TARGET _xmm POST_BUILD COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/src/move_files.sh)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/bin/xmm)
