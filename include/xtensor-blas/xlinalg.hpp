/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XLINALG_HPP
#define XLINALG_HPP

#include <algorithm>
#include <sstream>

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"

#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlapack.hpp"

namespace xt
{
namespace linalg
{

    template <class E1>
    typename E1::value_type norm(const xexpression<E1>& a, int ord = 2)
    {
        if (ord == 1)
        {
            return blas::asum(a);
        }
        else if (ord == 2)
        {
            return blas::nrm2(a);
        }
        std::stringstream ss;
        ss << "Norm " << ord << " not implemented!" << std::endl;
        throw std::runtime_error(ss.str());
    }

    template <class E1, class E2>
    auto solve(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        return lapack::gesv(a, b);
    }

    template <class E1>
    auto inverse(const xexpression<E1>& a)
    {
        // copy otherwise A gets overwritten
        auto a_c = copy_to_layout<layout_type::column_major>(a.derived_cast());
        return lapack::getri(a_c);
    }

    template <class E1>
    auto eig(const xexpression<E1>& A)
    {
        auto geev_res = lapack::geev(A);

        using value_type = typename E1::value_type;
        const auto N = A.shape()[0];

        xtensor<std::complex<value_type>, 1> eig_vals;
        eig_vals.reshape({N});
        xtensor<std::complex<value_type>, 2> eig_vecs;
        eig_vecs.reshape({N, N});

        xt::real(eig_vals) = std::get<geev_res>(0);
        xt::imag(eig_vals) = std::get<geev_res>(1);

        auto& VR = std::get<geev_res>(3);
        auto& wi = std::get<geev_res>(1);

        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N - 1; ++j)
            {
                if (wi[j] != 0)
                {
                    eig_vecs(i, j)     = std::complex<value_type>(VR(i, j),  VR(i, j + 1));
                    eig_vecs(i, j + 1) = std::complex<value_type>(VR(i, j), -VR(i, j + 1));
                    ++j;
                }
                else
                {
                    eig_vecs(i, j) = std::complex<value_type>(VR(i, j), 0);
                }
            }
        }

    }

    template <class T, class O>
    typename select_xtype<T, O>::type
    dot(const T& t, const O& o) {
        if (t.dimension() == 1 && o.dimension() == 1)
        {
            return blas::dot(t, o);
        }
        else
        {
            if (t.dimension() == 2 && o.dimension() == 1)
            {
                return blas::gemv(t, o);
            }
            else if (t.dimension() == 1 && o.dimension() == 2)
            {
                return blas::gemv(o, t, true);
            }
            else if (t.dimension() == 2 && o.dimension() == 2)
            {
                return blas::gemm(o, t);
            }
        }
        throw std::exception();
    }

    template <class E1, class E2>
    auto cross(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        using return_type = xtensor<typename E1::value_type, 1>;
        return_type res(typename return_type::shape_type{3});
        const E1& da = a.derived_cast();
        const E2& db = b.derived_cast();

        if (da.size() == 3 && db.size() == 3)
        {
            res(0) = da(1) * db(2) - da(2) * db(1);
            res(1) = da(2) * db(0) - da(0) * db(2);
            res(2) = da(0) * db(1) - da(1) * db(0);
        }
        else if (da.size() == 2 && db.size() == 3)
        {
            res(0) =   da(1) * db(2);
            res(1) = -(da(0) * db(2));
            res(2) =   da(0) * db(1) - da(1) * db(0);
        }
        else if (da.size() == 3 && db.size() == 2)
        {
            res(0) = -(da(2) * db(1));
            res(1) =   da(2) * db(0);
            res(2) =   da(0) * db(1) - da(1) * db(0);
        }
        else if (da.size() == 2 && db.size() == 2)
        {
            res(0) = 0;
            res(1) = 0;
            res(2) = da(0) * db(1) - da(1) * db(0);
        }
        else
        {
            throw std::exception();
        }
        return res;
    }
}
}
#endif