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
#include "xtensor/xcomplex.hpp"
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlapack.hpp"
#include "xtensor-blas/xlinalg.hpp"

using namespace std::complex_literals;

namespace xt
{
    TEST(xlinalg, matrixpower)
    {
        xarray<double> t1arg_0 = {{0,1,2},
                                  {3,4,5},
                                  {6,7,8}};

        auto t1res = xt::linalg::matrix_power(t1arg_0, 2);
        xarray<long> t1expected = {{ 15, 18, 21},
                                   { 42, 54, 66},
                                   { 69, 90,111}};
        EXPECT_TRUE(allclose(t1res, t1expected));

        auto t2res = xt::linalg::matrix_power(t1arg_0, 5);
        xarray<long> t2expected = {{ 32400, 41796, 51192},
                                   { 99468,128304,157140},
                                   {166536,214812,263088}};
        EXPECT_TRUE(allclose(t2res, t2expected));

        auto t3res = xt::linalg::matrix_power(t1arg_0, 41);
        xarray<double> t3expected = {{  1.06199622e+45,  1.36986674e+45,  1.67773727e+45},
                                     {  3.26000325e+45,  4.20507151e+45,  5.15013977e+45},
                                     {  5.45801029e+45,  7.04027628e+45,  8.62254226e+45}};
        EXPECT_TRUE(allclose(t3res, t3expected));

        xarray<double> t4arg_0 = {{-2., 1., 3.},
                                  { 3., 2., 1.},
                                  { 1., 2., 5.}};

        auto t4res = xt::linalg::matrix_power(t4arg_0, -2);
        xarray<double> t4expected = {{ 0.09259259,-0.09259259, 0.01851852},
                                     { 0.35185185, 0.64814815,-0.46296296},
                                     {-0.2037037 ,-0.2962963 , 0.25925926}};
        EXPECT_TRUE(allclose(t4res, t4expected));

        auto t5res = xt::linalg::matrix_power(t4arg_0, -13);
        xarray<double> t5expected = {{-0.02119919,-0.02993041, 0.02400524},
                                     { 0.15202629, 0.21469317,-0.17217602},
                                     {-0.0726041 ,-0.10253451, 0.08222825}};
        EXPECT_TRUE(allclose(t5res, t5expected));
    }

    TEST(xlinalg, det)
    {
        xarray<double> a = {{1,2}, {3,4}};
        double da = linalg::det(a);
        EXPECT_EQ(da, -2.0);
        xarray<double> b = {{0, 1,2}, {3,4, 5}, {6,7,8}};
        double db = linalg::det(b);
        EXPECT_EQ(db, 0.0);
        xarray<double> c = {{12, 1,2}, {3,4, 5}, {6,7,8}};
        double dc = linalg::det(c);
        EXPECT_NEAR(dc, -36, 1e-06);
    }
}
