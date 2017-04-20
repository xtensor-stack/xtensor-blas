/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlapack.hpp"
#include "xtensor-blas/xlinalg.hpp"


namespace xt
{
    TEST(xblas, matrix_times_vector)
    {
        xarray<double> m1
          {{1, 2, 3},
           {4, 5, 6}};
        xarray<double> b = {1, 2, 3};

        auto res = linalg::dot(m1, b);
        xarray<double> expected = {14, 32};
        EXPECT_EQ(expected, res);

        xarray<double> next_row = {{7, 8, 9}};
        auto res2 = linalg::dot(concatenate(xtuple(m1, next_row)), b);
        xarray<double> expected2 = {14, 32, 50};
        EXPECT_EQ(expected2, res2);
    }

    TEST(xblas, view_dot)
    {
        xarray<double> a = {1,2,3,4,5};
        xarray<double> b = {5,4,3,2,1};
        auto res = linalg::dot(a, b);

        xarray<double> expected = {35};
        EXPECT_EQ(expected, res);

        xarray<double> m1
          {{1, 2, 3},
           {4, 5, 6}};

        xarray<double> c = {1, 2};
        auto res2 = xt::linalg::dot(xt::view(m1, xt::all(), 1), c);
        xarray<double> expected2 = {12};
        EXPECT_EQ(expected2, res2);
    }

    TEST(xblas, norm)
    {
        auto a = linalg::norm(xt::arange<double>(15), 1);
        auto b = linalg::norm(xt::arange<double>(15), 2);
        xarray<double> c = {6,4,2,1};
        auto res = linalg::norm(c);

        EXPECT_EQ(a, 105.0);
        EXPECT_NEAR(b, 31.859064644147981, 1e-6);
        EXPECT_NEAR(res, 7.5498344352707498, 1e-6);
    }

    TEST(xblas, outer)
    {
        xarray<double> a = {1, 1, 1};

        xarray<double> b = arange(0, 3);

        xarray<long> expected = {{0,1,2},
                                 {0,1,2},
                                 {0,1,2}};

        auto t = linalg::outer(a, b);
        auto t2 = linalg::outer(a, xt::arange(0, 3));
        auto t3 = linalg::outer(xt::ones<double>({3}), xt::arange(0, 3));

        EXPECT_TRUE(all(equal(expected, t)));
        EXPECT_TRUE(all(equal(expected, t2)));
        EXPECT_TRUE(all(equal(expected, t3)));
    }
}