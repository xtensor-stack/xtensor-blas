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

    /**
     * Calculate 1- and 2-norm of vector.
     *
     * @param vec input vector
     * @param ord order of norm (1 or 2)
     * @return scalar result
     *
     * @tparam type of xexpression
     */
    template <class E1>
    typename E1::value_type norm(const xexpression<E1>& vec, int ord = 2)
    {
        if (ord == 1)
        {
            return blas::asum(vec);
        }
        else if (ord == 2)
        {
            return blas::nrm2(vec);
        }
        std::stringstream ss;
        ss << "Norm " << ord << " not implemented!" << std::endl;
        throw std::runtime_error(ss.str());
    }

    /**
     * Solve a linear matrix equation, or system of linear scalar equations.
     * Computes the “exact” solution, x, of the well-determined, i.e., full rank, 
     * linear matrix equation ax = b.
     * 
     * @param a Coefficient matrix
     * @param b Ordinate or “dependent variable” values.
     * @return Solution to the system a x = b. Returned shape is identical to b.
     */
    template <class E1, class E2>
    auto solve(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        return lapack::gesv(a, b);
    }

    /**
     * Compute the (multiplicative) inverse of a matrix.
     *
     * @param a Matrix to be inverted
     * @return (Multiplicative) inverse of the matrix a.
     */
    template <class E1>
    auto inv(const xexpression<E1>& a)
    {
        // copy otherwise A gets overwritten
        auto a_c = copy_to_layout<layout_type::column_major>(a.derived_cast());
        return lapack::getri(a_c);
    }

    /**
     * Compute the eigenvalues and right eigenvectors of a square array.
     *
     * @param Matrix for which the eigenvalues and right eigenvectors will be computed
     * @return std::tuple(w, v). The first element corresponds to the eigenvalues, 
     *                    each repeated according to its multiplicity. The eigenvalues 
     *                    are not necessarily ordered.
     *                    The second (1) element are the normalized (unit “length”) eigenvectors,
     *                    such that the column v[:, i] corresponds to the eigenvalue w[i].
     */
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
        return std::make_tuple(VR, eig_vecs);
    }

    /**
     * Non-broadcasting dot function.
     * In the case of two 1D vectors, computes the vector dot 
     * product. In the case of complex vectors, computes the dot 
     * product without conjugating the first argument.
     * If \em t or \em o is a 2D matrix, computes the matrix-times-vector
     * product. If both \em t and \em o ar 2D matrices, computes
     * the matrix-product.
     * 
     * @param t input array
     * @param o input array
     *
     * @return resulting array
     */
    template <class T, class O>
    auto dot(const T& t, const O& o) {
        if (t.dimension() == 1 && o.dimension() == 1)
        {
            if (is_complex<typename T::value_type>::value)
            {
                return blas::dotu(t, o);
            }
            else
            {
                return blas::dot(t, o);
            }
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

    /**
     * Computes the dot product for two vectors. 
     * Behaves different from \ref dot in the case of complex
     * vectors. If vectors are complex, vdot conjugates the first
     * argument \em t.
     * Note: Unlike NumPy, xtensor-blas currently doesn't flatten 
     * the input arguments.
     * 
     * @param t input vector (1D)
     * @param o input vector (1D)
     *
     * @return resulting array
     */
    template <class T, class O>
    auto vdot(const T& t, const O& o) {
        XTENSOR_ASSERT(t.dimension() == 1);
        XTENSOR_ASSERT(o.dimension() == 1);

        if (is_complex<typename T::value_type>::value)
        {
            return blas::dot(t, o);
        }
        else
        {
            return blas::dotu(t, o);
        }
    }

    /**
     * Compute the outer product of two vectors.
     * 
     * @param t input vector (1D)
     * @param o input vector (1D)
     *
     * @return resulting array
     */
    template <class T, class O>
    auto outer(const T& t, const O& o) {
        XTENSOR_ASSERT(t.dimension() == 1);
        XTENSOR_ASSERT(o.dimension() == 1);
        return blas::ger(t, o);
    }

    /**
     * Calculate matrix power A**n
     *
     * @param mat  The matrix
     * @param n    The exponent
     *
     * @return resulting array
     */
    template <class E>
    E matrix_power(E mat, int n)
    {
        XTENSOR_ASSERT(mat.dimension() == 2);
        XTENSOR_ASSERT(mat.shape()[0] == mat.shape()[1]);

        using xtype = E;
        xtype res(mat.shape());
        if (n == 0)
        {
            res = eye(mat.shape());
            return res;
        }
        else if (n < 0)
        {
            mat = inv(mat);
            n = -n;
        }

        res = mat;

        if (n <= 3)
        {
            for (int i = 0; i < n - 1; ++i)
            {
                res = blas::gemm(res, mat);
            }
            return res;
        }

        int i = 0;

        int bits, var = n;
        for(bits = 0; var != 0; ++bits)
        {
            var >>= 1;
        }

        while (~n & (1 << i))
        {
            mat = blas::gemm(mat, mat);
            ++i;
        }
        res = mat;
        ++i;
        for (; i < bits; ++i)
        {
            mat = blas::gemm(mat, mat);
            if (n & (1 << i))
            {
                res = blas::gemm(res, mat);
            }
        }
        return res;
    }

    /**
     * Non-broadcasting cross product between two vectors
     * Calculate cross product between two 1D vectors with 2- or 3 entries.
     * If only two entries are available, the third entry is assumed to be 0.
     * 
     * @param a input vector
     * @param b input vector
     * @return resulting array
     */
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
            throw std::runtime_error("a or b did not have appropriate size 2 or 3.");
        }
        return res;
    }
}
}
#endif