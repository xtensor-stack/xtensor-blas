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

    TEST(xblas, solve)
    {
        xarray<double> a = {{ 2, 1, 1},
                            {-1, 1,-1},
                            { 1, 2, 3}};

        xarray<double> vec = {2, 3, -10};
        xarray<double> expected = {3, 1, -5};

        auto res = linalg::solve(a, vec);
        EXPECT_EQ(expected, res);

        vec.reshape({3, 1});
        expected.reshape({3, 1});
        auto res2 = linalg::solve(a, vec);
        EXPECT_EQ(expected, res2);

        xarray<double> vec2 = {6, 2, -10};
        vec2.reshape({3, 1});

        auto res3 = linalg::solve(a, concatenate(xtuple(vec, vec2 * 3), 1));
        xarray<double> expected3 = {{ 3, 16},
                                    { 1,  4},
                                    {-5,-18}};
        EXPECT_EQ(expected3, res3);
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

    TEST(xblas, inverse)
    {
        xarray<double> a = {{ 2, 1, 1},
                            {-1, 1,-1},
                            { 1, 2, 3}};

        xarray<double> b = {{ 1, 0, 0},
                            { 0, 1, 0},
                            { 0, 0, 1}};

        auto t = linalg::inverse(a);

        xarray<double> expected = 
              {{ 0.55555556, -0.11111111, -0.22222222},
               { 0.22222222,  0.55555556,  0.11111111},
               {-0.33333333, -0.33333333,  0.33333333}};

        EXPECT_NEAR(expected(0, 0), t(0, 0), 1e-5);
        EXPECT_NEAR(expected(1, 0), t(1, 0), 1e-5);
        EXPECT_NEAR(expected(2, 0), t(2, 0), 1e-5);
        EXPECT_NEAR(expected(2, 1), t(2, 1), 1e-5);

        auto br = linalg::inverse(b);
        EXPECT_EQ(b, br);
        xarray<double> t_r_major(std::vector<std::size_t>{3, 3});
        assign_data(t_r_major, t, true);
        auto almost_eye = linalg::dot(t_r_major, a);
        auto e = xt::eye(3);
        auto d = almost_eye - e;
        auto min = xt::amin(d);
        EXPECT_NEAR(min(), 0.0, 1e-6);
    }
}