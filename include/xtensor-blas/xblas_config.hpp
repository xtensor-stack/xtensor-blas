/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_CONFIG_HPP
#define XBLAS_CONFIG_HPP

#define XTENSOR_BLAS_VERSION_MAJOR 0
#define XTENSOR_BLAS_VERSION_MINOR 1
#define XTENSOR_BLAS_VERSION_PATCH 0

#ifndef USE_CXXLAPACK
#define USE_CXXLAPACK
#endif

#ifndef BLAS_IDX
#define BLAS_IDX int
#endif

namespace xt
{
    using XBLAS_INDEX = int;
}

#endif
