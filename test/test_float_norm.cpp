/***************************************************************************
 * Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

// this test is for https://github.com/xtensor-stack/xtensor-blas/issues/206

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"

#include "gtest/gtest.h"
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlinalg.hpp"

namespace xt
{

    TEST(xblas, norm_complex_float)
    {
        xt::xarray<std::complex<float>> a = {std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f)};
        auto res = linalg::norm(a);

        EXPECT_NEAR(res.real(), 5.4772f, 1e-3f);
        EXPECT_NEAR(res.imag(), 0.0f, 1e-3f);
    }

    TEST(xblas, norm_float_arange)
    {
        xt::linalg::norm(xt::arange<float>(15), 1);
    }

}  // namespace xt
