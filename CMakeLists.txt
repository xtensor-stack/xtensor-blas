############################################################################
# Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor-blas)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FLENS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/xtensor-blas/flens)
set(XTENSOR_BLAS_INCLUDE_DIR ${INCLUDE_DIR} ${FLENS_INCLUDE_DIR})

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

find_package(xtensor REQUIRED)
message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xensor")

# Build
# =====

set(XTENSOR_BLAS_HEADERS
    ${INCLUDE_DIR}/xtensor-blas/xblas.hpp
    ${INCLUDE_DIR}/xtensor-blas/xblas_utils.hpp
    ${INCLUDE_DIR}/xtensor-blas/xblas_config.hpp
    ${INCLUDE_DIR}/xtensor-blas/xlapack.hpp
    ${INCLUDE_DIR}/xtensor-blas/xlinalg.hpp
)

OPTION(CXXBLAS_DEBUG "print cxxblas debug information" OFF)
OPTION(HAVE_CBLAS "use cblas" ON)

if(HAVE_CBLAS)
  add_definitions(-DHAVE_CBLAS=1)
endif()

if (CXXBLAS_DEBUG)
  add_definitions(-DCXXBLAS_DEBUG=1)
endif()

OPTION(BUILD_TESTS "xtensor-blas test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    include_directories(${XTENSOR_BLAS_INCLUDE_DIR})
    include_directories(${xtensor_INCLUDE_DIRS})
    add_subdirectory(test)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${XTENSOR_BLAS_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-blas)

install(DIRECTORY ${INCLUDE_DIR}/xtensor-blas/flens/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(XTENSOR_BLAS_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for xtensor-blasConfig.cmake")

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
