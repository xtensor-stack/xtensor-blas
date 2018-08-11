/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
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

#include "xtensor/xio.hpp"

using namespace std::complex_literals;

namespace xt
{
    TEST(xlapack, eigenvalues)
    {
        xarray<double> eig_arg_0 = {{ 0.89342434, 0.96630682, 0.83113658, 0.9014204 , 0.17622395},
                                    { 0.01114647, 0.93096724, 0.35509599, 0.35329223, 0.65759337},
                                    { 0.27868701, 0.376794  , 0.63310696, 0.90892131, 0.35454718},
                                    { 0.02962539, 0.20561053, 0.2004051 , 0.83641883, 0.08335324},
                                    { 0.76958296, 0.23132089, 0.33539779, 0.70616527, 0.40256713}};
        auto eig_res = xt::linalg::eig(eig_arg_0);

        xtensor<std::complex<double>, 1> eig_expected_0 = { 2.24745601+0.i        , 0.24898158+0.51158566i, 0.24898158-0.51158566i,
                                                            0.66252212+0.i        , 0.28854321+0.i        };

        xtensor<std::complex<double>, 2> eig_expected_1 = {{-0.67843725+0.i        ,-0.00104977+0.50731553i,-0.00104977-0.50731553i,
                                                            -0.48456457+0.i        ,-0.11153304+0.i        },
                                                           {-0.38393722+0.i        ,-0.42892828-0.30675499i,-0.42892828+0.30675499i,
                                                            -0.60497432+0.i        ,-0.55233486+0.i        },
                                                           {-0.39453548+0.i        , 0.10153693-0.12657944i, 0.10153693+0.12657944i,
                                                             0.35111489+0.i        , 0.80267297+0.i        },
                                                           {-0.15349367+0.i        ,-0.04903747+0.08226059i,-0.04903747-0.08226059i,
                                                             0.48726345+0.i        ,-0.10533951+0.i        },
                                                           {-0.46162383+0.i        , 0.65501769+0.i        , 0.65501769-0.i        ,
                                                            -0.19620376+0.i        , 0.16463982+0.i        }};
        xarray<std::complex<double>> eigvals = std::get<0>(eig_res);
        xarray<std::complex<double>> eigvecs = std::get<1>(eig_res);

        EXPECT_TRUE(allclose(xt::imag(eigvals), xt::imag(eig_expected_0)));
        EXPECT_TRUE(allclose(xt::real(eigvals), xt::real(eig_expected_0)));
        EXPECT_TRUE(allclose(abs(imag(eigvecs)), abs(imag(eig_expected_1))));
        EXPECT_TRUE(allclose(abs(real(eigvecs)), abs(real(eig_expected_1))));
    }

    TEST(xlapack, inverse)
    {
        xarray<double> a = {{ 2, 1, 1},
                            {-1, 1,-1},
                            { 1, 2, 3}};

        xarray<double> b = {{ 1, 0, 0},
                            { 0, 1, 0},
                            { 0, 0, 1}};

        auto t = linalg::inv(a);

        xarray<double, layout_type::column_major> expected = 
              {{ 0.55555556, -0.11111111, -0.22222222},
               { 0.22222222,  0.55555556,  0.11111111},
               {-0.33333333, -0.33333333,  0.33333333}};

        EXPECT_TRUE(allclose(expected, t));

        auto br = linalg::inv(b);
        EXPECT_EQ(b, br);
        auto t_r_major = xarray<double>::from_shape({3, 3});
        assign_data(t_r_major, t, true);
        auto almost_eye = linalg::dot(t_r_major, a);
        auto e = xt::eye(3);
        auto d = almost_eye - e;
        auto min = xt::amin(d);
        EXPECT_NEAR(min(), 0.0, 1e-6);
    }

    TEST(xlapack, single_element_inverse)
    {
        xtensor<double, 2> a = xt::ones<double>({1, 1});
        auto res = linalg::inv(a);
        EXPECT_EQ(res(), 1.);
    }

    TEST(xlapack, solve)
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

}
