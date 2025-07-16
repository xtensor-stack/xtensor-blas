############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.29)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-blas-test)

    enable_testing()

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
    find_package(xtensor-blas REQUIRED CONFIG)
    set(XTENSOR_BLAS_INCLUDE_DIR ${xblas_INCLUDE_DIRS})
endif ()

find_package(doctest REQUIRED)
find_package(Threads)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(set_compiler_flag.cmake)

if(CPP23)
  # User requested C++20, but compiler might not oblige.
  set_compiler_flag(
    _cxx_std_flag CXX
    "-std=c++23"  # this should work with GNU, Intel, PGI
    "/std:c++23"  # this should work with MSVC
  )
  if(_cxx_std_flag)
    message(STATUS "Building with C++23")
  endif()
else()
  set_compiler_flag(
    _cxx_std_flag CXX REQUIRED
    "-std=c++20"  # this should work with GNU, Intel, PGI
    "/std:c++20"  # this should work with MSVC
  )
  message(STATUS "Building with C++20")
endif()

if(NOT _cxx_std_flag)
  message(FATAL_ERROR "xtensor-blas needs a C++20-compliant compiler.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
  CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
  if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /EHsc /MP /bigobj /wd4800")
  set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT WIN32)
    CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
    if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
  else() # We are using clang-cl
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /EHsc /MP /bigobj -Wno-unused-command-line-argument")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
  endif()
else()
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${XBLAS_INCLUDE_DIR})

if(USE_OPENBLAS)
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
    test_lstsq.cpp
    test_qr.cpp
    test_dot.cpp
    test_tensordot.cpp
    test_lstsq.cpp
    test_dot.cpp
    test_dot_extended.cpp
    test_qr.cpp
    test_float_norm.cpp
)

add_executable(test_xtensor_blas ${XTENSOR_BLAS_TESTS} ${XTENSOR_BLAS_HEADERS} ${XTENSOR_HEADERS})
target_link_libraries(test_xtensor_blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} doctest::doctest ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND test_xtensor_blas DEPENDS test_xtensor_blas)
add_test(NAME xtest COMMAND test_xtensor_blas)
