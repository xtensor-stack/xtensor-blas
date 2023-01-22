############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor-blas)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(XTENSOR_BLAS_INCLUDE_DIR ${INCLUDE_DIR})

# Configuration
# =============

# TODO: use the library directory of the found BLAS implementation instead of
# CMAKE_INSTALL_LIBDIR
set(XTENSOR_BLAS_CLING_LIBRARY_DIR_64 "\"${CMAKE_INSTALL_PREFIX}/lib64\"")
set(XTENSOR_BLAS_CLING_LIBRARY_DIR_32 "\"${CMAKE_INSTALL_PREFIX}/lib32\"")
set(XTENSOR_BLAS_CLING_LIBRARY_DIR "\"${CMAKE_INSTALL_PREFIX}/lib\"")

configure_file (
    "${INCLUDE_DIR}/xtensor-blas/xblas_config_cling.hpp.in"
    "${INCLUDE_DIR}/xtensor-blas/xblas_config_cling.hpp"
)

# Versionning
# ===========

file(STRINGS "${INCLUDE_DIR}/xtensor-blas/xblas_config.hpp" xtensor_blas_version_defines
     REGEX "#define XTENSOR_BLAS_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_blas_version_defines})
    if(ver MATCHES "#define XTENSOR_BLAS_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_BLAS_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XTENSOR_BLAS_VERSION_MAJOR}.${XTENSOR_BLAS_VERSION_MINOR}.${XTENSOR_BLAS_VERSION_PATCH})
message(STATUS "xtensor-blas v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

set(xtensor_REQUIRED_VERSION 0.24.0)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

# Build
# =====

set(XTENSOR_BLAS_HEADERS
    ${INCLUDE_DIR}/xtensor-blas/xblas.hpp
    ${INCLUDE_DIR}/xtensor-blas/xblas_utils.hpp
    ${INCLUDE_DIR}/xtensor-blas/xblas_config.hpp
    ${INCLUDE_DIR}/xtensor-blas/xblas_config_cling.hpp
    ${INCLUDE_DIR}/xtensor-blas/xlapack.hpp
    ${INCLUDE_DIR}/xtensor-blas/xlinalg.hpp
)

add_library(xtensor-blas INTERFACE)

target_include_directories(xtensor-blas
  INTERFACE
    $<BUILD_INTERFACE:${XTENSOR_BLAS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

OPTION(CXXBLAS_DEBUG "print cxxblas debug information" OFF)
OPTION(CPP17 "enables C++17" OFF)
OPTION(XTENSOR_USE_FLENS_BLAS "use FLENS generic implementation instead of cblas" OFF)
# Decide whether to use OpenBLAS or not.
# The user might have the folder containing OpenBLASConfig.cmake file
# in the paths inspected by CMake already and just setting this to ON
# would be enough to detect OpenBLAS.
# If that is not the case, one can pass OpenBLAS_DIR without the boolean
# toggle.
OPTION(USE_OPENBLAS "use OpenBLAS (requires suitable OpenBLASConfig.cmake)" OFF)
if(OpenBLAS_DIR)
  set(USE_OPENBLAS ON)
endif()

if(XTENSOR_USE_FLENS_BLAS)
  add_definitions(-DXTENSOR_USE_FLENS_BLAS=1)
endif()

if (CXXBLAS_DEBUG)
  add_definitions(-DCXXBLAS_DEBUG=1)
endif()

OPTION(BUILD_TESTS "xtensor-blas test suite" OFF)
OPTION(BUILD_BENCHMARK "xtensor-blas test suite" OFF)

OPTION(DOWNLOAD_GTEST "download gtest and build from source" OFF)
OPTION(DOWNLOAD_GBENCHMARK "download google benchmark and build from source" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    enable_testing()
    include_directories(${XTENSOR_BLAS_INCLUDE_DIR})
    include_directories(${xtensor_INCLUDE_DIRS})
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARKS)
    include_directories(${XTENSOR_BLAS_INCLUDE_DIR})
    include_directories(${xtensor_INCLUDE_DIRS})
    add_subdirectory(benchmark)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xtensor-blas
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_BLAS_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-blas)

install(DIRECTORY ${INCLUDE_DIR}/xflens
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PATTERN *.tgz EXCLUDE
        PATTERN Makefile EXCLUDE
        PATTERN dummy.in.cc EXCLUDE
        PATTERN filter.pm EXCLUDE
        PATTERN CMakeLists.txt EXCLUDE)

set(XTENSOR_BLAS_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensor-blasConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_BLAS_CMAKECONFIG_INSTALL_DIR})

# xtensor is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xblasConfigVersion.cmake so that an xblasConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_BLAS_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_BLAS_CMAKECONFIG_INSTALL_DIR})

install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_BLAS_CMAKECONFIG_INSTALL_DIR})
