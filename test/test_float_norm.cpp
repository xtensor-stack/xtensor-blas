/***************************************************************************
 * Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

// this test is for https://github.com/xtensor-stack/xtensor-blas/issues/206

#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "xtensor/views/xview.hpp"

#include "doctest/doctest.h"
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlinalg.hpp"

namespace xt
{

    TEST_SUITE("xblas")
    {
        TEST_CASE("norm_complex_float")
        {
            xt::xarray<std::complex<float>> a = {std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f)};
            auto res = linalg::norm(a);

            CHECK(res.real() == doctest::Approx(5.4772f).epsilon(1e-3f));
            CHECK(res.imag() == doctest::Approx(0.0f).epsilon(1e-3f));
        }

        TEST_CASE("norm_float_arange")
        {
            xt::linalg::norm(xt::arange<float>(15), 1);
        }
    }
}  // namespace xt
