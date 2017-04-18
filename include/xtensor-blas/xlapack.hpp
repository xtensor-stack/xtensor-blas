/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XLAPACK_HPP
#define XLAPACK_HPP

#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"

#include "xtensor-blas/xblas_utils.hpp"
#include "xtensor-blas/xblas_config.hpp"

#ifndef USE_CXXLAPACK
#define USE_CXXLAPACK
#endif
#include "thirdparty/FLENS/flens/flens.cxx"

namespace xt
{

namespace lapack
{
    template <class E1, class E2>
    xarray<typename E1::value_type> gesv(const xexpression<E1>& A, const xexpression<E2>& B)
    {
        auto&& da = view_eval<layout_type::column_major>(A.derived_cast());
        auto&& db = view_eval<layout_type::column_major>(B.derived_cast());

        XTENSOR_ASSERT(da.dimension() == 2);
        XTENSOR_ASSERT(da.layout() == layout_type::column_major);

        using result_type = xarray<typename E1::value_type, layout_type::column_major>;
        typename result_type::shape_type s = {da.shape()[0]};

        std::vector<XBLAS_INDEX> piv(da.shape()[0]);

        cxxlapack::gesv<XBLAS_INDEX>(
            (XBLAS_INDEX) da.shape()[0], 
            db.dimension() > 1 ? (XBLAS_INDEX) db.shape().back() : 1,
            da.raw_data(),
            (XBLAS_INDEX) da.strides()[1],
            piv.data(),
            db.raw_data(), 
            !db.strides().back() < db.shape().front() ? (XBLAS_INDEX) db.shape().front() : (XBLAS_INDEX) db.strides().back()
        );

        return db;
    }

    template <class E1>
    auto getri(const xexpression<E1>& A)
    {
        using value_type = typename E1::value_type;

        auto dA = A.derived_cast();

        XTENSOR_ASSERT(dA.dimension() == 2);
        XTENSOR_ASSERT(dA.layout() == xt::layout_type::column_major);

        std::vector<XBLAS_INDEX> piv(dA.shape()[0]);

        cxxlapack::getrf<XBLAS_INDEX>(
            (XBLAS_INDEX) dA.shape()[0], 
            (XBLAS_INDEX) dA.shape()[1], 
            dA.raw_data(),
            (XBLAS_INDEX) dA.strides().back(), 
            piv.data()
        );

        std::vector<value_type> work(1);

        // get work size
        cxxlapack::getri<XBLAS_INDEX>(
            (XBLAS_INDEX) dA.shape()[0],
            dA.raw_data(),
            (XBLAS_INDEX) dA.strides().back(),
            piv.data(),
            work.data(), -1
        );

        work.resize(std::size_t(work[0]));

        cxxlapack::getri<XBLAS_INDEX>(
            (XBLAS_INDEX) dA.shape()[0],
            dA.raw_data(),
            (XBLAS_INDEX) dA.strides().back(),
            piv.data(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return dA;
    }

    template <class E1>
    auto geev(const xexpression<E1>& A)
    {
        // TODO implement for complex numbers

        auto&& dA = view_eval<layout_type::column_major>(A.derived_cast());
        XTENSOR_ASSERT(dA.dimension() == 2);
        XTENSOR_ASSERT(dA.layout() == layout_type::column_major);

        using value_type = typename E1::value_type;
        using xtype = xtensor<value_type, 2, layout_type::column_major>;

        const auto N = dA.shape()[0];
        std::vector<value_type> work(1);

        xtensor<value_type, 1> wr;
        wr.reshape({N});
        xtensor<value_type, 1> wi;
        wi.reshape({N});

        xtensor<value_type, 2, layout_type::column_major> VL(dA.shape());
        xtensor<value_type, 2, layout_type::column_major> VR(dA.shape());

        cxxlapack::geev<int>(
            'N',
            'V',
            (XBLAS_INDEX) N,
            dA.raw_data(),
            (XBLAS_INDEX) dA.strides().back(),
            wr.raw_data(),
            wi.raw_data(),
            VL.raw_data(), 
            (XBLAS_INDEX) VL.strides().back(),
            VR.raw_data(), 
            (XBLAS_INDEX) VR.strides().back(),
            work.data(), -1
        );

        work.resize(std::size_t(work[0]));

        cxxlapack::geev<int>(
            'N', // don't compute left eig vectors
            'V', // compute right eigenvectors
            N,
            dA.raw_data(), 
            (XBLAS_INDEX) dA.strides().back(),
            wr.raw_data(),
            wi.raw_data(),
            VL.raw_data(), 
            (XBLAS_INDEX) VL.strides().back(),
            VR.raw_data(),
            (XBLAS_INDEX) VR.strides().back(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return std::make_tuple(wr, wi, VL, VR);
    }
}

}
#endif