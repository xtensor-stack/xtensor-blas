/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_HPP
#define XBLAS_HPP

#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"

#include "xtensor-blas/xblas_utils.hpp"

#ifndef USE_CXXLAPACK
#define USE_CXXLAPACK
#endif
#include "thirdparty/FLENS/flens/flens.cxx"


namespace xt
{

namespace blas
{
    /**
     * Calculate the dot product between two vectors
     * @param a vector of n elements
     * @param b vector of n elements
     * @returns scalar result
     */
    template <class E1, class E2>
    xtensor<typename E1::value_type, 1> dot(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        xtensor<typename E1::value_type, 1> res({0});

        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());
        auto&& bd = view_eval<E1::static_layout>(b.derived_cast());

        cxxblas::dot(ad.size(),
                     ad.raw_data() + ad.raw_data_offset(), ad.strides().front(),
                     bd.raw_data() + bd.raw_data_offset(), bd.strides().front(),
                     res(0));
        return res;
    }

    /**
     * Calculate the 1-norm of a vector
     * @param a vector of n elements
     * @returns scalar result
     */
    template <class E1>
    typename E1::value_type asum(const xexpression<E1>& a)
    {
        typename E1::value_type res;
        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());
        cxxblas::asum(ad.size(),
                      ad.raw_data() + ad.raw_data_offset(), ad.strides().front(), 
                      res);
        return res;
    }

    /**
     * Calculate the 2-norm of a vector
     * @param a vector of n elements
     * @returns scalar result
     */
    template <class E1>
    typename E1::value_type nrm2(const xexpression<E1>& a)
    {
        typename E1::value_type res;
        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());

        cxxblas::nrm2(ad.size(),
                      ad.raw_data() + ad.raw_data_offset(), ad.strides().front(), 
                      res);
        return res;
    }

    template <class E1, class E2>
    xt::xarray<typename E1::value_type> gemv(const xexpression<E1>& A, const xexpression<E2>& x, 
                                             bool transpose = false,
                                             const xscalar<typename E1::value_type> alpha = 1)
    {
        // General Matrix times vector calculatess
        // y := alpha * A * x + beta * y

        auto&& dA = view_eval<E1::static_layout>(A.derived_cast());
        auto&& dx = view_eval<E2::static_layout>(x.derived_cast());

        using result_type = typename select_xtype<E1, E2>::type;
        typename result_type::shape_type result_shape({dA.shape()[0]});

        if (transpose)
        {
            std::reverse(result_shape.begin(), result_shape.end());
        }

        result_type res(result_shape);


        cxxblas::gemv(
            get_blas_storage_order(dA), 
            transpose ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans, 
            dA.shape()[0], dA.shape()[1],
            alpha(), // alpha
            dA.raw_data(), dA.strides().front(),
            dx.raw_data(), dx.strides().front(), 
            0.f, // beta 
            res.raw_data(), 1ul
        );

        return res;
    }

    /**
     * Calculate the matrix-matrix product of matrix @A and matrix @B
     * @param A matrix of m-by-n elements
     * @param B matrix of n-by-k elements
     * @returns matrix of m-by-k elements
     */
    template <class E1, class E2>
    xarray<typename E1::value_type> gemm(const xexpression<E1>& A, const xexpression<E2>& B,
                                         const xscalar<typename E1::value_type> alpha = 1,
                                         const xscalar<typename E1::value_type> beta = 0,
                                         bool transpose_A = false, bool transpose_B = false)
    {
        auto&& da = view_eval(A.derived_cast());
        auto&& db = view_eval(B.derived_cast());

        XTENSOR_ASSERT(da.layout() == db.layout());

        using return_type = typename select_xtype<E1, E2>::type;
        typename return_type::shape_type s = {
          transpose_A ? da.shape()[1] : da.shape()[0],
          transpose_B ? db.shape()[0] : db.shape()[1],
        };

        return_type res(s);

        cxxblas::gemm(
            get_blas_storage_order(da), 
            transpose_A ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans, 
            transpose_B ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans, 
            da.shape()[0], da.shape()[1], db.shape()[0],
            alpha(),
            da.raw_data(), da.strides().front(),
            db.raw_data(), db.strides().front(), 
            beta(),
            res.raw_data(), res.strides().front()
        );

        return res;
    }

    template <class E1, class E2>
    xarray<typename E1::value_type> ger(const xexpression<E1>& x, const xexpression<E2>& y,
                                        const xscalar<typename E1::value_type> alpha = 1)
    {
        // outer product
        // A := alpha * x * y' + A
        auto&& dx = view_eval(x.derived_cast());
        auto&& dy = view_eval(y.derived_cast());

        XTENSOR_ASSERT(dx.dimension() == 1);
        XTENSOR_ASSERT(dy.dimension() == 1);

        using return_type = typename select_xtype<E1, E2>::type;
        typename return_type::shape_type s = {dx.shape()[0], dy.shape()[0]};
        return_type res(s);

        cxxblas::ger(
            get_blas_storage_order(res), 
            dx.shape()[0],
            dy.shape()[0],
            alpha(),
            dx.raw_data(), dx.strides().front(),
            dy.raw_data(), dy.strides().front(), 
            res.raw_data(), res.strides().front()
        );

        return res;
    }
}
}
#endif