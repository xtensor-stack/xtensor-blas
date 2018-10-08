#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstrided_view.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xt
{
    TEST(xtensordot, outer_product)
    {
      xarray<double> a = xt::ones<double>({3,3,3});
      xarray<double> b = xt::ones<double>({2,2}) * 5.0;
      xarray<double> e1 = xt::ones<double>({3,3,3,2,2}) * 5.0;

      auto r1 = linalg::tensordot(a, b, 0);
      EXPECT_EQ(e1, r1);
    }

    TEST(xtensordot, outer_product_cm)
    {
      xarray<float, layout_type::column_major> a = xt::ones<float>({3,3,3});
      xarray<float, layout_type::column_major> b = xt::ones<float>({2,2}) * 5.0;
      xarray<float, layout_type::column_major> e1 = xt::ones<float>({3,3,3,2,2}) * 5.0;

      auto r1 = linalg::tensordot(a, b, 0);
      EXPECT_EQ(e1, r1);

    }

    TEST(xtensordot, outer_product_mixed_layout)
    {
      xarray<float, layout_type::column_major> a = xt::ones<float>({3,3,3});
      xarray<float> b = xt::ones<float>({2,2}) * 5.0;
      xarray<float, layout_type::column_major> e1 = xt::ones<float>({3,3,3,2,2}) * 5.0;

      auto r1 = linalg::tensordot(a, b, 0);
      EXPECT_EQ(e1, r1);

      xarray<float> e2 = xt::ones<float>({2,2,3,3,3}) * 5.0;
      auto r2 = linalg::tensordot(b, a, 0);
      EXPECT_EQ(e2, r2);

    }

    TEST(xtensordot, inner_product)
    {
      xarray<double> a = xt::ones<double>({3,3,2,2});
      xarray<double> b = xt::ones<double>({2,2,10});
      auto r1 = linalg::tensordot(a, b);
      EXPECT_TRUE(all(equal(r1, 4)));
      EXPECT_TRUE(r1.shape().size() == 3);
      EXPECT_TRUE(r1.shape()[0] == 3);
      EXPECT_TRUE(r1.shape()[1] == 3);
      EXPECT_TRUE(r1.shape()[2] == 10);

      EXPECT_THROW(linalg::tensordot(a, b, 3), std::runtime_error);
      EXPECT_THROW(linalg::tensordot(b, a), std::runtime_error);

    }

    TEST(xtensordot, inner_product_cm)
    {
      xarray<double, layout_type::column_major> a = xt::ones<double>({3,3,2,2});
      xarray<double, layout_type::column_major> b = xt::ones<double>({2,2,10});
      auto r1 = linalg::tensordot(a, b);
      EXPECT_TRUE(all(equal(r1, 4)));
      EXPECT_TRUE(r1.shape().size() == 3);
      EXPECT_TRUE(r1.shape()[0] == 3);
      EXPECT_TRUE(r1.shape()[1] == 3);
      EXPECT_TRUE(r1.shape()[2] == 10);

      EXPECT_THROW(linalg::tensordot(a, b, 3), std::runtime_error);
      EXPECT_THROW(linalg::tensordot(b, a), std::runtime_error);

    }

    TEST(xtensordot, inner_product_mixed_layout)
    {
      xarray<double> a = xt::ones<double>({3,3,2,2});
      xarray<double, layout_type::column_major> b = xt::ones<double>({3,2,2,10});
      auto r1 = linalg::tensordot(a, b, 3);
      EXPECT_TRUE(all(equal(r1, 12.0)));
      EXPECT_TRUE(r1.shape().size() == 2);
      EXPECT_TRUE(r1.shape()[0] == 3);
      EXPECT_TRUE(r1.shape()[1] == 10);

      EXPECT_THROW(linalg::tensordot(b, a), std::runtime_error);

    }

    TEST(xtensordot, tuple_ax)
    {
      xarray<double> a = {{{{0, 1},
                            {2, 3},
                            {4, 5}},
                           {{6, 7},
                            {8, 9},
                            {10, 11}}},
                          {{{12, 13},
                            {14,15},
                            {16,17}},
                           {{18, 19},
                            {20, 21},
                            {22,23}}},
                          {{{24,25},
                            {26,27},
                            {28,29}},
                           {{30,31},
                            {32,33},
                            {34,35}}}};
      xarray<double> b = xt::ones<double>({2, 3, 2, 3});
      auto r1 = linalg::tensordot(a, b, {1, 3, 2}, {0, 2, 1});
      xarray<double> e1 = {{66, 66, 66},
                           {210,210,210},
                           {354, 354, 354}};
      EXPECT_EQ(r1, e1);
      auto r2 = linalg::tensordot(a, b, {1, 3, 2, 0}, {0, 2, 1, 3});
      xarray<double> e2 = xarray<double>::from_shape({1,1});
      e2(0,0) = 630;
      EXPECT_EQ(r2(0,0), e2(0,0));
    }

    TEST(xtensordot, tuple_ax_cm)
    {
      xarray<double, layout_type::column_major> a = {{{{0, 1},
                            {2, 3},
                            {4, 5}},
                           {{6, 7},
                            {8, 9},
                            {10, 11}}},
                          {{{12, 13},
                            {14,15},
                            {16,17}},
                           {{18, 19},
                            {20, 21},
                            {22,23}}},
                          {{{24,25},
                            {26,27},
                            {28,29}},
                           {{30,31},
                            {32,33},
                            {34,35}}}};
      xarray<double, layout_type::column_major> b = xt::ones<double>({2, 3, 2, 3});
      auto r1 = linalg::tensordot(a, b, {1, 3, 2}, {0, 2, 1});
      xarray<double, layout_type::column_major> e1 = {{66, 66, 66},
                           {210,210,210},
                           {354, 354, 354}};
      EXPECT_EQ(r1, e1);
      auto r2 = linalg::tensordot(a, b, {1, 3, 2, 0}, {0, 2, 1, 3});
      xarray<double, layout_type::column_major> e2 = xarray<double>::from_shape({1,1});
      e2(0,0) = 630;
      EXPECT_EQ(r2(0,0), e2(0,0));

    }

    TEST(xtensordot, tuple_ax_mixed_layout)
    {
      xarray<double, layout_type::column_major> a = {{{{0, 1},
                            {2, 3},
                            {4, 5}},
                           {{6, 7},
                            {8, 9},
                            {10, 11}}},
                          {{{12, 13},
                            {14,15},
                            {16,17}},
                           {{18, 19},
                            {20, 21},
                            {22,23}}},
                          {{{24,25},
                            {26,27},
                            {28,29}},
                           {{30,31},
                            {32,33},
                            {34,35}}}};
      xarray<double> b = xt::ones<double>({2, 3, 2, 3});
      auto r1 = linalg::tensordot(a, b, {1, 3, 2}, {0, 2, 1});
      xarray<double, layout_type::column_major> e1 = {{66, 66, 66},
                           {210,210,210},
                           {354, 354, 354}};
      EXPECT_EQ(r1, e1);
      auto r2 = linalg::tensordot(a, b, {1, 3, 2, 0}, {0, 2, 1, 3});
      xarray<double, layout_type::column_major> e2 = xarray<double>::from_shape({1,1});
      e2(0,0) = 630;
      EXPECT_EQ(r2(0,0), e2(0,0));
    }
}
