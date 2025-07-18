/***************************************************************************
 * Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "xtensor/views/xview.hpp"

#include "doctest/doctest.h"
#include "xtensor-blas/xlinalg.hpp"

namespace xt
{
    TEST_SUITE("xtensor_dot")
    {
        TEST_CASE("outer_product")
        {
            xarray<double> a = xt::ones<double>({3, 3, 3});
            xarray<double> b = xt::ones<double>({2, 2}) * 5.0;
            xarray<double> e1 = xt::ones<double>({3, 3, 3, 2, 2}) * 5.0;

            auto r1 = linalg::tensordot(a, b, 0);
            CHECK_EQ(e1, r1);
        }

        TEST_CASE("outer_product_cm")
        {
            xarray<float, layout_type::column_major> a = xt::ones<float>({3, 3, 3});
            xarray<float, layout_type::column_major> b = xt::ones<float>({2, 2}) * 5.0;
            xarray<float, layout_type::column_major> e1 = xt::ones<float>({3, 3, 3, 2, 2}) * 5.0;

            auto r1 = linalg::tensordot(a, b, 0);
            CHECK_EQ(e1, r1);
        }

        TEST_CASE("outer_product_mixed_layout")
        {
            xarray<float, layout_type::column_major> a = xt::ones<float>({3, 3, 3});
            xarray<float> b = xt::ones<float>({2, 2}) * 5.0;
            xarray<float, layout_type::column_major> e1 = xt::ones<float>({3, 3, 3, 2, 2}) * 5.0;

            auto r1 = linalg::tensordot(a, b, 0);
            CHECK_EQ(e1, r1);

            xarray<float> e2 = xt::ones<float>({2, 2, 3, 3, 3}) * 5.0;
            auto r2 = linalg::tensordot(b, a, 0);
            CHECK_EQ(e2, r2);
        }

        TEST_CASE("inner_product")
        {
            xarray<double> a = xt::ones<double>({3, 3, 2, 2});
            xarray<double> b = xt::ones<double>({2, 2, 10});
            auto r1 = linalg::tensordot(a, b);
            CHECK(all(equal(r1, 4)));
            CHECK(r1.shape().size() == 3);
            CHECK(r1.shape()[0] == 3);
            CHECK(r1.shape()[1] == 3);
            CHECK(r1.shape()[2] == 10);

            CHECK_THROWS_AS(linalg::tensordot(a, b, 3), std::runtime_error);
            CHECK_THROWS_AS(linalg::tensordot(b, a), std::runtime_error);
        }

        TEST_CASE("inner_product_cm")
        {
            xarray<double, layout_type::column_major> a = xt::ones<double>({3, 3, 2, 2});
            xarray<double, layout_type::column_major> b = xt::ones<double>({2, 2, 10});
            auto r1 = linalg::tensordot(a, b);
            CHECK(all(equal(r1, 4)));
            CHECK(r1.shape().size() == 3);
            CHECK(r1.shape()[0] == 3);
            CHECK(r1.shape()[1] == 3);
            CHECK(r1.shape()[2] == 10);

            CHECK_THROWS_AS(linalg::tensordot(a, b, 3), std::runtime_error);
            CHECK_THROWS_AS(linalg::tensordot(b, a), std::runtime_error);
        }

        TEST_CASE("inner_product_mixed_layout")
        {
            xarray<double> a = xt::ones<double>({3, 3, 2, 2});
            xarray<double, layout_type::column_major> b = xt::ones<double>({3, 2, 2, 10});
            auto r1 = linalg::tensordot(a, b, 3);
            CHECK(all(equal(r1, 12.0)));
            CHECK(r1.shape().size() == 2);
            CHECK(r1.shape()[0] == 3);
            CHECK(r1.shape()[1] == 10);

            CHECK_THROWS_AS(linalg::tensordot(b, a), std::runtime_error);
        }

        TEST_CASE("tuple_ax")
        {
            xarray<double> a = {
                {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}},
                {{{12, 13}, {14, 15}, {16, 17}}, {{18, 19}, {20, 21}, {22, 23}}},
                {{{24, 25}, {26, 27}, {28, 29}}, {{30, 31}, {32, 33}, {34, 35}}}
            };
            xarray<double> b = xt::ones<double>({2, 3, 2, 3});
            auto r1 = linalg::tensordot(a, b, {1, 3, 2}, {0, 2, 1});
            xarray<double> e1 = {{66, 66, 66}, {210, 210, 210}, {354, 354, 354}};
            CHECK_EQ(r1, e1);
            auto r2 = linalg::tensordot(a, b, {1, 3, 2, 0}, {0, 2, 1, 3});
            xarray<double> e2 = xarray<double>::from_shape({1, 1});
            e2(0, 0) = 630;
            CHECK_EQ(r2(0, 0), e2(0, 0));
        }

        TEST_CASE("tuple_ax_cm")
        {
            xarray<double, layout_type::column_major> a = {
                {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}},
                {{{12, 13}, {14, 15}, {16, 17}}, {{18, 19}, {20, 21}, {22, 23}}},
                {{{24, 25}, {26, 27}, {28, 29}}, {{30, 31}, {32, 33}, {34, 35}}}
            };
            xarray<double, layout_type::column_major> b = xt::ones<double>({2, 3, 2, 3});
            auto r1 = linalg::tensordot(a, b, {1, 3, 2}, {0, 2, 1});
            xarray<double, layout_type::column_major> e1 = {{66, 66, 66}, {210, 210, 210}, {354, 354, 354}};
            CHECK_EQ(r1, e1);
            auto r2 = linalg::tensordot(a, b, {1, 3, 2, 0}, {0, 2, 1, 3});
            xarray<double, layout_type::column_major> e2 = xarray<double>::from_shape({1, 1});
            e2(0, 0) = 630;
            CHECK_EQ(r2(0, 0), e2(0, 0));
        }

        TEST_CASE("tuple_ax_mixed_layout")
        {
            xarray<double, layout_type::column_major> a = {
                {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}},
                {{{12, 13}, {14, 15}, {16, 17}}, {{18, 19}, {20, 21}, {22, 23}}},
                {{{24, 25}, {26, 27}, {28, 29}}, {{30, 31}, {32, 33}, {34, 35}}}
            };
            xarray<double> b = xt::ones<double>({2, 3, 2, 3});
            auto r1 = linalg::tensordot(a, b, {1, 3, 2}, {0, 2, 1});
            xarray<double, layout_type::column_major> e1 = {{66, 66, 66}, {210, 210, 210}, {354, 354, 354}};
            CHECK_EQ(r1, e1);

            auto r2 = linalg::tensordot(a, b, {1, 3, 2, 0}, {0, 2, 1, 3});
            xarray<double, layout_type::column_major> e2 = {630};

            CHECK_EQ(r2, e2);
        }

        TEST_CASE("view")
        {
            xarray<int> a = reshape_view(arange<int>(3 * 2 * 3 * 2), {3, 2, 3, 2});
            xarray<int> b = reshape_view(arange<int>(3 * 3 * 2 * 2), {3, 3, 2, 2});

            xarray<int> e1 = {{34, 90, 146}, {46, 134, 222}, {58, 178, 298}};

            auto res1 = linalg::tensordot(
                view(a, 0, all(), all(), all()),
                view(b, 0, all(), all(), all()),
                {0, 2},
                {1, 2}
            );

            CHECK_EQ(res1, e1);
            CHECK_EQ(res1.dimension(), 2u);
            CHECK_EQ(res1.shape()[0], 3u);
            CHECK_EQ(res1.shape()[1], 3u);
        }

        TEST_CASE("strided_view_range")
        {
            xarray<int> a = reshape_view(arange<int>(3 * 2 * 3 * 2), {3, 2, 3, 2});
            xarray<int> b = reshape_view(arange<int>(3 * 3 * 2 * 2), {3, 3, 2, 2});

            xarray<int> e1 = {{1064, 1144}, {1136, 1224}};

            auto res1 = linalg::tensordot(
                strided_view(a, {range(0, 2), all(), range(0, 2), all()}),
                strided_view(b, {range(0, 2), range(0, 2), all(), all()}),
                {0, 1, 2},
                {0, 1, 2}
            );
            CHECK_EQ(res1, e1);
            CHECK_EQ(res1.dimension(), 2u);
            CHECK_EQ(res1.shape()[0], 2u);
            CHECK_EQ(res1.shape()[1], 2u);
        }

        TEST_CASE("reducing_dim_view")
        {
            xarray<int> a = reshape_view(arange<int>(3 * 2 * 3 * 2), {3, 2, 3, 2});
            xarray<int> b = reshape_view(arange<int>(3 * 3 * 2 * 2), {3, 3, 2, 2});

            xarray<int> e = {1589};
            auto r = linalg::tensordot(view(a, 0, 1, all(), all()), view(b, 2, all(), 1, all()));
            CHECK_EQ(r, e);
        }

        TEST_CASE("reducing_dim_strided_view")
        {
            xarray<int> a = reshape_view(arange<int>(3 * 2 * 3 * 2), {3, 2, 3, 2});
            xarray<int> b = reshape_view(arange<int>(3 * 3 * 2 * 2), {3, 3, 2, 2});

            xarray<int> e = {1589};
            auto r = linalg::tensordot(
                strided_view(a, {0, 1, all(), all()}),
                strided_view(b, {2, all(), 1, all()})
            );
            CHECK_EQ(r, e);
        }
    }
}  // namespace xt
