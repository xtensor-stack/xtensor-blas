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

#include "flens/cxxlapack/cxxlapack.cxx"

#include "xtensor-blas/xblas_config.hpp"
#include "xtensor-blas/xblas_utils.hpp"

namespace xt
{

namespace lapack
{
    /**
     * Interface to LAPACK gesv.
     *
     * @param A matrix in
     * @param b vector (right hand side)
     *
     * @return solution to ``Ax = b``
     */
    template <class E1, class E2>
    int gesv(E1& A, E2& b)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);
        XTENSOR_ASSERT(b.dimension() <= 2);
        XTENSOR_ASSERT(b.layout() == layout_type::column_major);

        std::vector<XBLAS_INDEX> piv(A.shape()[0]);

        XBLAS_INDEX b_dim = b.dimension() > 1 ? (XBLAS_INDEX) b.shape().back() : 1;
        XBLAS_INDEX b_stride = b_dim == 1 ? (XBLAS_INDEX) b.shape().front() : (XBLAS_INDEX) b.strides().back();

        int info = cxxlapack::gesv<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            b_dim,
            A.raw_data(),
            (XBLAS_INDEX) A.strides()[1],
            piv.data(),
            b.raw_data(),
            b_stride
        );

        return info;
    }

    template <class E1>
    auto getrf(E1& A)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        std::vector<XBLAS_INDEX> piv(std::min(A.shape()[0], A.shape()[1]));

        int info = cxxlapack::getrf<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            piv.data()
        );

        return std::make_tuple(info, std::move(piv));
    }

    template <class E1>
    inline auto orgqr(E1& A, std::vector<typename E1::value_type>& tau)
    {
        using value_type = typename E1::value_type;

        std::vector<value_type> work(1);

        int info = cxxlapack::orgqr<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            (XBLAS_INDEX) tau.size(),
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            tau.data(),
            work.data(),
            (XBLAS_INDEX) -1
        );

        if (info != 0)
        {
            throw std::runtime_error("Could not find workspace size for orgqr.");
        }

        work.resize(work[0]);

        info = cxxlapack::orgqr<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            (XBLAS_INDEX) tau.size(),
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            tau.data(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return info;
    }

    template <class E1>
    inline auto ungqr(E1& A, std::vector<typename E1::value_type>& tau)
    {
        using value_type = typename E1::value_type;

        std::vector<value_type> work(1);

        int info = cxxlapack::ungqr<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            (XBLAS_INDEX) tau.size(),
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            tau.data(),
            work.data(),
            (XBLAS_INDEX) -1
        );

        if (info != 0)
        {
            throw std::runtime_error("Could not find workspace size for ungqr.");
        }

        work.resize((std::size_t) std::real(work[0]));

        info = cxxlapack::ungqr<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            (XBLAS_INDEX) tau.size(),
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            tau.data(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return info;
    }

    template <class E1>
    auto geqrf(E1& A)
    {
        using value_type = typename E1::value_type;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        std::vector<value_type> tau(std::min(A.shape()[0], A.shape()[1]));

        std::vector<value_type> work(1);

        int info = cxxlapack::geqrf<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            tau.data(),
            work.data(),
            (XBLAS_INDEX) -1
        );

        if (info != 0)
        {
            throw std::runtime_error("Could not find workspace size for geqrf.");
        }

        work.resize((std::size_t) std::real(work[0]));

        info = cxxlapack::geqrf<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            tau.data(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return std::make_tuple(info, tau);
    }

    template <class E1>
    auto gesdd(E1& A, char jobz = 'A')
    {
        using value_type = typename E1::value_type;
        using xtype1 = xtensor<value_type, 1, layout_type::column_major>;
        using xtype2 = xtensor<value_type, 2, layout_type::column_major>;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        std::vector<value_type> work(1);

        std::size_t m = A.shape()[0];
        std::size_t n = A.shape()[1];

        xtype1 s;
        s.reshape({ std::max(1ul, std::min(m, n)) });

        xtype2 u, vt;
        XBLAS_INDEX u_stride = 1, vt_stride = 1;
        if (jobz == 'A' || (jobz == 'O' && m < n))
        {
            u.reshape({n, n});
            vt.reshape({n, n});
            u_stride = (XBLAS_INDEX) u.strides().back();
            vt_stride = (XBLAS_INDEX) vt.strides().back();
        }
        if (jobz == 'S')
        {
            u.reshape({m, std::min(m, n)});
            vt.reshape({m, std::min(m, n)});
            u_stride = (XBLAS_INDEX) u.strides().back();
            vt_stride = (XBLAS_INDEX) vt.strides().back();
        }

        std::vector<XBLAS_INDEX> iwork(8 * std::min(m, n));

        int info = cxxlapack::gesdd<XBLAS_INDEX>(
            jobz,
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            s.raw_data(),
            u.raw_data(),
            u_stride,
            vt.raw_data(),
            vt_stride,
            work.data(),
            (XBLAS_INDEX) -1,
            iwork.data()
        );

        if (info != 0)
        {
            throw std::runtime_error("Could not find workspace size for gesdd.");
        }

        work.resize((std::size_t) std::real(work[0]));

        info = cxxlapack::gesdd<XBLAS_INDEX>(
            jobz,
            (XBLAS_INDEX) A.shape()[0],
            (XBLAS_INDEX) A.shape()[1],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            s.raw_data(),
            u.raw_data(),
            u_stride,
            vt.raw_data(),
            vt_stride,
            work.data(),
            (XBLAS_INDEX) work.size(),
            iwork.data()
        );

        return std::make_tuple(info, u, s, vt);
    }


    template <class E1>
    auto potr(E1& A, char uplo = 'L')
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        int info = cxxlapack::potrf<XBLAS_INDEX>(
            uplo,
            (XBLAS_INDEX) A.shape()[0],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back()
        );

        return info;
    }

    /**
     * Interface to LAPACK getri.
     *
     * @param A matrix to invert
     * @return inverse of A
     */
    template <class E1>
    int getri(E1& A, std::vector<XBLAS_INDEX>& piv)
    {
        using value_type = typename E1::value_type;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        std::vector<value_type> work(1);

        // get work size
        int info = cxxlapack::getri<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            piv.data(),
            work.data(),
            -1
        );

        if (info > 0)
        {
            throw std::runtime_error("Could not find workspace size for getri.");
        }

        work.resize(std::size_t(work[0]));

        info = cxxlapack::getri<XBLAS_INDEX>(
            (XBLAS_INDEX) A.shape()[0],
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            piv.data(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return info;
    }

    /**
     * Interface to LAPACK geev.
     *
     * @param A matrix for which eigenvalues are to be calculated
     * @return tuple of (wr, wi, VR, VL) where wr and wi are the real and imaginary
     *         part of the eigenvalue, VR are the right eigenvectors (the only ones computed)
     *         and VL is currently meaningless. Please consult a LAPACK documentation
     *         to find out how VR is structured.
     */
    template <class E1>
    auto geev(E1& A, char jobvl = 'N', char jobvr = 'V')
    {
        // TODO implement for complex numbers

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        using value_type = typename E1::value_type;
        using xtype = xtensor<value_type, 2, layout_type::column_major>;

        const auto N = A.shape()[0];
        std::vector<value_type> work(1);

        std::vector<std::size_t> vN = {N};
        xarray<value_type, layout_type::column_major> wr(vN);
        xarray<value_type, layout_type::column_major> wi(vN);

        typename xtype::shape_type shp({A.shape()[0], A.shape()[1]});
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        int info = cxxlapack::geev<XBLAS_INDEX>(
            jobvl,
            jobvr,
            (XBLAS_INDEX) N,
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            wr.raw_data(),
            wi.raw_data(),
            VL.raw_data(),
            (XBLAS_INDEX) VL.strides().back(),
            VR.raw_data(),
            (XBLAS_INDEX) VR.strides().back(),
            work.data(),
            -1
        );

        if (info != 0)
        {
            throw std::runtime_error("Could not find workspace size for geev.");
        }

        work.resize(std::size_t(work[0]));

        info = cxxlapack::geev<XBLAS_INDEX>(
            jobvl,
            jobvr,
            (XBLAS_INDEX) N,
            A.raw_data(),
            (XBLAS_INDEX) A.strides().back(),
            wr.raw_data(),
            wi.raw_data(),
            VL.raw_data(),
            (XBLAS_INDEX) VL.strides().back(),
            VR.raw_data(),
            (XBLAS_INDEX) VR.strides().back(),
            work.data(),
            (XBLAS_INDEX) work.size()
        );

        return std::make_tuple(info, wr, wi, VL, VR);
    }
}

}
#endif