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

#include "xtensor-blas/xblas_config.hpp"
#include "xtensor-blas/xblas_utils.hpp"

#include "flens/cxxblas/cxxblas.cxx"


namespace xt
{

namespace blas
{
    /**
     * Calculate the dot product between two vectors, conjugating 
     * the first argument \em a in the case of complex vectors.
     * 
     * @param a vector of n elements
     * @param b vector of n elements
     * @returns scalar result
     */
    template <class E1, class E2>
    xarray<typename E1::value_type> dot(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        xarray<typename E1::value_type> res({0});

        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());
        auto&& bd = view_eval<E1::static_layout>(b.derived_cast());

        cxxblas::dot<BLAS_IDX>(
            (BLAS_IDX) ad.size(),
            ad.raw_data() + ad.raw_data_offset(),
            (BLAS_IDX) ad.strides().front(),
            bd.raw_data() + bd.raw_data_offset(),
            (BLAS_IDX) bd.strides().front(),
            res(0)
        );
        return res;
    }

    /**
     * Calculate the dot product between two complex vectors, not conjugating the 
     * first argument \em a.
     * 
     * @param a vector of n elements
     * @param b vector of n elements
     * @returns scalar result
     */
    template <class E1, class E2>
    xarray<typename E1::value_type> dotu(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        xarray<typename E1::value_type> res({0});

        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());
        auto&& bd = view_eval<E1::static_layout>(b.derived_cast());

        cxxblas::dotu<BLAS_IDX>(
            (BLAS_IDX) ad.size(),
            ad.raw_data() + ad.raw_data_offset(),
            (BLAS_IDX) ad.strides().front(),
            bd.raw_data() + bd.raw_data_offset(),
            (BLAS_IDX) bd.strides().front(),
            res(0)
        );
        return res;
    }

    /**
     * Calculate the 1-norm of a vector
     * 
     * @param a vector of n elements
     * @returns scalar result
     */
    template <class E1>
    typename E1::value_type asum(const xexpression<E1>& a)
    {
        typename E1::value_type res;
        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());
        cxxblas::asum<BLAS_IDX>(
            (BLAS_IDX) ad.size(),
            ad.raw_data() + ad.raw_data_offset(),
            (BLAS_IDX) ad.strides().front(), 
            res
        );
        return res;
    }

    /**
     * Calculate the 2-norm of a vector
     * 
     * @param a vector of n elements
     * @returns scalar result
     */
    template <class E1>
    typename E1::value_type nrm2(const xexpression<E1>& a)
    {
        typename E1::value_type res;
        auto&& ad = view_eval<E1::static_layout>(a.derived_cast());

        cxxblas::nrm2<BLAS_IDX>(
            (BLAS_IDX) ad.size(),
            ad.raw_data() + ad.raw_data_offset(),
            (BLAS_IDX) ad.strides().front(), 
            res
        );
        return res;
    }

    /**
     * Calculate the general matrix times vector product according to 
     * ``y := alpha * A * x + beta * y``.
     *
     * @param A matrix of n x m elements
     * @param x vector of n elements
     * @param transpose select if A should be transposed
     * @param alpha scalar scale factor
     * @returns the resulting vector
     */
    template <class E1, class E2, class value_type = typename E1::value_type>
    xt::xarray<value_type> gemv(const xexpression<E1>& A, const xexpression<E2>& x, 
                                bool transpose = false,
                                const xscalar<value_type> alpha = value_type(1))
    {
        auto&& dA = view_eval<E1::static_layout>(A.derived_cast());
        auto&& dx = view_eval<E2::static_layout>(x.derived_cast());

        using result_type = typename select_xtype<E1, E2>::type;
        typename result_type::shape_type result_shape({dA.shape()[0]});

        if (transpose)
        {
            std::reverse(result_shape.begin(), result_shape.end());
        }

        result_type res(result_shape);  // TODO make double 

        cxxblas::gemv<BLAS_IDX>(
            get_blas_storage_order(dA), 
            transpose ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans, 
            (BLAS_IDX) dA.shape()[0],
            (BLAS_IDX) dA.shape()[1],
            alpha(), // alpha
            dA.raw_data(),
            (BLAS_IDX) dA.strides().front(),
            dx.raw_data(),
            (BLAS_IDX) dx.strides().front(), 
            value_type(0), // beta 
            res.raw_data(),
            (BLAS_IDX) 1
        );

        return res;
    }

    /**
     * Calculate the matrix-matrix product of matrix @A and matrix @B
     * @param A matrix of m-by-n elements
     * @param B matrix of n-by-k elements
     * @returns matrix of m-by-k elements
     */
    template <class E1, class E2, class value_type = typename E1::value_type>
    xarray<value_type> gemm(const xexpression<E1>& A, const xexpression<E2>& B,
                            const xscalar<value_type> alpha = value_type(1),
                            const xscalar<value_type> beta = value_type(0),
                            bool transpose_A = false, bool transpose_B = false)
    {
        auto&& da = view_eval(A.derived_cast());
        auto&& db = view_eval(B.derived_cast());

        XTENSOR_ASSERT(da.layout() == db.layout());
        XTENSOR_ASSERT(da.dimension() == 2);
        XTENSOR_ASSERT(db.dimension() == 2);

        using return_type = typename select_xtype<E1, E2>::type;
        typename return_type::shape_type s = {
          transpose_A ? da.shape()[1] : da.shape()[0],
          transpose_B ? db.shape()[0] : db.shape()[1],
        };

        return_type res(s);

        cxxblas::gemm<BLAS_IDX>(
            get_blas_storage_order(da), 
            transpose_A ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans, 
            transpose_B ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans, 
            (BLAS_IDX) da.shape()[0],
            (BLAS_IDX) da.shape()[1],
            (BLAS_IDX) db.shape()[0],
            alpha(),
            da.raw_data(),
            (BLAS_IDX) da.strides().front(),
            db.raw_data(),
            (BLAS_IDX) db.strides().front(), 
            beta(),
            res.raw_data(),
            (BLAS_IDX) res.strides().front()
        );

        return res;
    }

    /**
     * Calculate the outer product of vector x and y.
     * According to A:= alpha * x * y' + A
     *
     * @param x vector of n elements
     * @param y vector of m elements
     * @param alpha scalar scale factor
     * @returns matrix of n-by-m elements
     */
    template <class E1, class E2, class value_type = typename E1::value_type>
    xarray<value_type> ger(const xexpression<E1>& x, const xexpression<E2>& y,
                           const xscalar<value_type> alpha = 1)
    {
        // outer product
        // A := alpha * x * y' + A
        auto&& dx = view_eval(x.derived_cast());
        auto&& dy = view_eval(y.derived_cast());

        XTENSOR_ASSERT(dx.dimension() == 1);
        XTENSOR_ASSERT(dy.dimension() == 1);

        using return_type = typename select_xtype<E1, E2>::type;
        typename return_type::shape_type s = {dx.shape()[0], dy.shape()[0]};
        return_type res(s, 0);

        cxxblas::ger<BLAS_IDX>(
            get_blas_storage_order(res), 
            (BLAS_IDX) dx.shape()[0],
            (BLAS_IDX) dy.shape()[0],
            alpha(),
            dx.raw_data(),  (BLAS_IDX) dx.strides().front(),
            dy.raw_data(),  (BLAS_IDX) dy.strides().front(), 
            res.raw_data(), (BLAS_IDX) res.strides().front()
        );

        return res;
    }
}
}
#endif