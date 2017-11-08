#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstridedview.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xt
{
    TEST(xdot, matrix_times_vector)
    {
        xarray<float> a = xt::ones<float>({1, 4});
        xarray<float> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {1, 1, 1}};

        xarray<float> e1 = {{13, 16, 19}};

        auto r1 = linalg::dot(a, b);
        EXPECT_EQ(e1, r1);

        xarray<float> c = xt::ones<float>({3, 1});

        auto r2 = linalg::dot(b, c);
        xarray<float> e2 = {{6, 15, 24, 3}};
        e2.reshape({4, 1});
        EXPECT_EQ(e2, r2);

        EXPECT_THROW(linalg::dot(b, a), std::runtime_error);
        EXPECT_THROW(linalg::dot(c, b), std::runtime_error);
    }

    TEST(xdot, square_matrix_times_vector)
    {
        xarray<float> a = {{1, 1, 1}};
        xarray<float> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        auto r1 = linalg::dot(a, b);

        xarray<float> e1 = {{12, 15, 18}};
        EXPECT_EQ(r1, e1);

        auto r2 = linalg::dot(b, xt::transpose(a));
        xarray<float> e2 = xarray<float>::from_shape({3, 1});
        e2(0, 0) = 6.f;
        e2(1, 0) = 15.f;
        e2(2, 0) = 24.f;
        EXPECT_EQ(r2, e2);

        EXPECT_THROW(linalg::dot(b, a), std::runtime_error);
    }

    TEST(xdot, vector_times_vector)
    {
        xarray<float> a = xt::ones<float>({1, 3});
        xarray<float> b = xt::ones<float>({3, 1});

        auto r1 = linalg::dot(a, b);

        xarray<float> e1 = xarray<float>::from_shape({1, 1});
        e1(0, 0) = 3;

        EXPECT_EQ(e1, r1);

        auto r2 = linalg::dot(b, a);
        xarray<float> e2 = xt::ones<float>({3, 3});
        EXPECT_EQ(e2, r2);

        auto r3 = linalg::dot(b, e1);
        EXPECT_EQ(b * 3, r3);
    }

}