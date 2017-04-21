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

        xarray<std::complex<double>> arg_0 = {{ 0.95368636+0.32324664i, 0.49936872+0.22164004i, 0.30452434+0.78922905i},
                                              { 0.84118920+0.59652768i, 0.42052057+0.97211559i, 0.19916742+0.83068058i},
                                              { 0.67065616+0.56830636i, 0.00268706+0.29410473i, 0.69147455+0.7052149i }};
        auto res = linalg::det(arg_0);

        auto res_i = std::imag(res);
        auto res_r = std::real(res);
        EXPECT_NEAR(0.4201495908415372, res_i, 1e-06);
        EXPECT_NEAR(-0.07633013993862534, res_r, 1e-06);
    }

    TEST(xlinalg, slogdet)
    {
        xarray<std::complex<double>> arg_0 = {{ 0.13373658+0.43025551i, 0.42593478+0.17539337i, 0.18840853+0.24669458i},
                                              { 0.82800224+0.11797823i, 0.40310379+0.14037109i, 0.88204561+0.96870283i},
                                              { 0.35427657+0.1233739i , 0.22740960+0.94019582i, 0.05410180+0.86462543i}};
        auto resc = linalg::slogdet(arg_0);
        auto sc = std::get<0>(resc);
        auto sl = std::real(std::get<1>(resc));
        auto scr = std::real(sc);
        auto sci = std::imag(sc);

        EXPECT_NEAR(-0.8818794751329891, sl, 1e-06);
        EXPECT_NEAR(0.8473375077176295, scr, 1e-06);
        EXPECT_NEAR(0.5310547504870624, sci, 1e-06);

        xarray<double> arg_b = {{ 0.20009016, 0.33997118, 0.74433611},
                                { 0.52721448, 0.2449798 , 0.49085606},
                                { 0.49757477, 0.97304175, 0.05011255}};
        auto res = linalg::slogdet(arg_b);
        double expected_0 = 1.0;
        double expected_1 = -1.3017524147193602;
        auto sres = std::get<0>(res);
        auto lres = std::get<1>(res);
        EXPECT_EQ(expected_0, sres);
        EXPECT_NEAR(expected_1, lres, 1e-06);
    }

    TEST(xlinalg, svd)
    {
        xarray<double> arg_0 = {{0,1,2},
                                {3,4,5},
                                {6,7,8}};

        auto res = linalg::svd(arg_0);

        xarray<double, layout_type::column_major> expected_0 = {{-0.13511895, 0.90281571, 0.40824829},
                                                                {-0.49633514, 0.29493179,-0.81649658},
                                                                {-0.85755134,-0.31295213, 0.40824829}};
        xarray<double, layout_type::column_major> expected_1 = {  1.42267074e+01,  1.26522599e+00,  5.89938022e-16};
        xarray<double, layout_type::column_major> expected_2 = {{-0.4663281 ,-0.57099079,-0.67565348},
                                                                {-0.78477477,-0.08545673, 0.61386131},
                                                                {-0.40824829, 0.81649658,-0.40824829}};

        EXPECT_TRUE(allclose(std::get<0>(res), expected_0));
        EXPECT_TRUE(allclose(std::get<1>(res), expected_1));
        EXPECT_TRUE(allclose(std::get<2>(res), expected_2));
    }

    TEST(xlinalg, matrix_rank)
    {
        int a = linalg::matrix_rank(eye<double>(4));
        EXPECT_EQ(4, a);

        xarray<double> b = eye<double>(4);
        b(1, 1) = 0;
        int rb = linalg::matrix_rank(b);
        EXPECT_EQ(3, rb);
        int ro = linalg::matrix_rank(ones<double>({4, 4}));
        EXPECT_EQ(1, ro);
        int rz = linalg::matrix_rank(zeros<double>({4, 4}));
        EXPECT_EQ(0, rz);
    }
}
