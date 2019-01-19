############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-blas-test)

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIR})
    find_package(xtensor-blas REQUIRED CONFIG)
    set(XTENSOR_BLAS_INCLUDE_DIR ${xblas_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" HAS_CPP17_FLAG)

    if (CPP17 AND HAS_CPP17_FLAG)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "Building with -std=c++17")
    elseif (HAS_CPP14_FLAG)
        set(CMAKE_CXX_STANDARD 14)
        message(STATUS "Building with -std=c++14")
    else()
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++14 support!")
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    if(DOWNLOAD_GTEST)
        # Download and unpack googletest at configure time
        configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)
    else()
        # Copy local source of googletest at configure time
        configure_file(copyGTest.cmake.in googletest-download/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES gtest_main gtest)
else()
    find_package(GTest REQUIRED)
endif()

find_package(Threads)

include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)
include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${XBLAS_INCLUDE_DIR})

find_package(xtensor REQUIRED)
if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

set(XTENSOR_BLAS_TESTS
    main.cpp
    test_blas.cpp
    test_lapack.cpp
    test_linalg.cpp
    test_dot.cpp
    test_tensordot.cpp
)

add_executable(test_xtensor_blas ${XTENSOR_BLAS_TESTS} ${XTENSOR_BLAS_HEADERS} ${XTENSOR_HEADERS})
if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    add_dependencies(test_xtensor_blas gtest_main)
endif()

target_link_libraries(test_xtensor_blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND test_xtensor_blas DEPENDS test_xtensor_blas)
