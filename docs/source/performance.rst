.. Copyright (c) 2017, Wolf Vollprecht, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _perf-and-link:

Performance and Linking
=======================

For optimal performance, the target program **has** to  be linked against
a BLAS implementation. The BLAS implementation that we install by default
with conda is ``OpenBLAS``, but other options, such as ``MKL`` are available
on conda, too.
Additionally, the compile time define ``-DHAVE_CBLAS`` should be enabled,
otherwise FLENS will use the slower internal implementation.

In order to link against ``OpenBLAS`` from CMake, the following lines have
to be added to the ``CMakeLists.txt`` file.

.. code:: cmake

    add_definitions(-DHAVE_CBLAS=1)

    if (WIN32)
        find_package(OpenBLAS REQUIRED)
        set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
    else()
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()

    message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
    message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

    target_link_libraries(your_target_name ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


If CMake is not used, the flags can be passed manually to e.g. ``g++``:

.. code:: bash

    g++ test.cpp -o test -lblas -llapack -DHAVE_CBLAS=1

