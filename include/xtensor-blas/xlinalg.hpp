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
#include <limits>
#include <sstream>
#include <chrono>

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xstridedview.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"

#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlapack.hpp"
#include "xtensor-blas/xblas_utils.hpp"

namespace xt
{
namespace linalg
{

    enum normorder {
        frob,
        nuc,
        inf,
        neg_inf
    };

    /**
     * Calculate 1- and 2-norm of vector.
     *
     * @param vec input vector
     * @param ord order of norm (1 or 2)
     * @return scalar result
     *
     * @tparam type of xexpression
     */
    template <class E>
    typename E::value_type norm(const xexpression<E>& vec, int ord)
    {
        using value_type = typename E::value_type;

        const auto& v = vec.derived_cast();

        value_type result = 0;
        if (v.dimension() == 1)
        {
            if (ord == 1)
            {
                blas::asum(v, result);
                return result;
            }
            else if (ord == 2)
            {
                blas::nrm2(v, result);
                return result;
            }
            else if (ord == 0)
            {
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    result += (v(i) != 0);
                }
                return result;
            }
            else
            {
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    result += std::abs(std::pow(v(i), ord));
                }
                return std::pow(result, 1./ (double) ord);
            }
        }
        else if (v.dimension() == 2)
        {
            if (ord == 1 || ord == -1)
            {
                xtensor<value_type, 1> s = xt::sum(xt::abs(v), {0});
                if (ord == 1)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
            if (ord == 2 || ord == -2)
            {
                auto M = copy_to_layout<layout_type::column_major>(v);
                auto gesdd_res = lapack::gesdd(M, 'N');
                auto& s = std::get<2>(gesdd_res);
                if (ord == 2)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
        }
        std::stringstream ss;
        ss << "Norm " << ord << " not implemented!" << std::endl;
        throw std::runtime_error(ss.str());
    }

    template <class E>
    typename E::value_type norm(const xexpression<E>& vec, normorder ord)
    {
        using value_type = typename E::value_type;
        const auto& v = vec.derived_cast();
        if (v.dimension() == 2)
        {
            if (ord == normorder::frob)
            {
                return std::sqrt(xt::sum(xt::pow(xt::abs(v), 2))());
            }
            if (ord == normorder::nuc)
            {
                auto M = copy_to_layout<layout_type::column_major>(v);
                auto gesdd_res = lapack::gesdd(M, 'N');
                auto& s = std::get<2>(gesdd_res);
                return std::accumulate(s.begin(), s.end(), value_type(0));
            }
            if (ord == normorder::inf || ord == normorder::neg_inf)
            {
                xtensor<value_type, 1> s = xt::sum(abs(v), {1});
                if (ord == normorder::inf)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
        }
        std::stringstream ss;
        ss << "Norm " << ord << " not implemented!" << std::endl;
        throw std::runtime_error(ss.str());
    }

    template <class E>
    typename E::value_type norm(const xexpression<E>& vec)
    {
        const auto& v = vec.derived_cast();
        if (v.dimension() == 1)
        {
            return norm(vec, 2);
        }
        else
        {
            return norm(vec, normorder::frob);
        }
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
    auto solve(const xexpression<E1>& A, const xexpression<E2>& b)
    {
        auto dA = copy_to_layout<layout_type::column_major>(A.derived_cast());
        auto db = copy_to_layout<layout_type::column_major>(b.derived_cast());

        int info = lapack::gesv(dA, db);

        if (info != 0)
        {
            throw std::runtime_error("The solution could not be computed");
        }

        return db;
    }

    /**
     * Compute the (multiplicative) inverse of a matrix.
     *
     * @param a Matrix to be inverted
     * @return (Multiplicative) inverse of the matrix a.
     */
    template <class E1>
    auto inv(const xexpression<E1>& A)
    {
        auto dA = copy_to_layout<layout_type::column_major>(A.derived_cast());

        uvector<XBLAS_INDEX> piv(std::min(dA.shape()[0], dA.shape()[1]));

        int info = lapack::getrf(dA, piv);
        if (info != 0)
        {
            throw std::runtime_error("getrf failed");
        }

        info = lapack::getri(dA, piv);
        if (info > 0)
        {
            throw std::runtime_error("Singular matrix not invertible (getri).");
        }
        return dA;
    }

    /**
     * Calculate the condition number of matrix M
     */
    template <class E>
    auto cond(const xexpression<E>& M, int ord)
    {
        return norm(M, ord) * norm(inv(M), ord);
    }

    template <class E>
    auto cond(const xexpression<E>& M, normorder ord)
    {
        return norm(M, ord) * norm(inv(M), ord);
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
    template <class E, std::enable_if_t<!is_complex<typename E::value_type>::value>* = nullptr>
    auto eig(const xexpression<E>& A)
    {
        using underlying_type = typename E::value_type;
        using value_type = typename E::value_type;

        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> wr(vN);
        xtensor<value_type, 1, layout_type::column_major> wi(vN);

        std::array<std::size_t, 2> shp({N, N});
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        // jobvl N: left eigenvectors of A are not computed
        // jobvr V: then right eigenvectors of A are computed
        int info = lapack::geev(M, 'N', 'V', wr, wi, VL, VR);

        if (info != 0)
        {
            throw std::runtime_error("Eigenvalue calculation did not converge.");
        }

        xtensor<std::complex<underlying_type>, 2> eig_vecs;
        eig_vecs.reshape({N, N});
        xtensor<std::complex<underlying_type>, 1> eig_vals;
        eig_vals.reshape({N});

        xt::real(eig_vals) = wr;
        xt::imag(eig_vals) = wi;

        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
            {
                if (wi(j) != 0)
                {
                    eig_vecs(i, j)     = std::complex<underlying_type>(VR(i, j),  VR(i, j + 1));
                    eig_vecs(i, j + 1) = std::complex<underlying_type>(VR(i, j), -VR(i, j + 1));
                    ++j;
                }
                else
                {
                    eig_vecs(i, j) = std::complex<underlying_type>(VR(i, j), 0);
                }
            }
        }

        return std::make_tuple(eig_vals, eig_vecs);
    }

    template <class E, std::enable_if_t<is_complex<typename E::value_type>::value>* = nullptr>
    auto eig(const xexpression<E>& A)
    {
        using value_type = typename E::value_type;
        using underlying_type = typename value_type::value_type;

        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        std::array<std::size_t, 2> shp({N, N});
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        // jobvl N: left eigenvectors of A are not computed
        // jobvr V: then right eigenvectors of A are computed
        int info = lapack::geev(M, 'N', 'V', w, VL, VR);

        if (info != 0)
        {
            throw std::runtime_error("Eigenvalue calculation did not converge.");
        }

        return std::make_tuple(w, VR);
    }

    /**
     * Compute the eigenvalues of a square xexpression.
     *
     * @param Matrix for which the eigenvalues and right eigenvectors will be computed
     * @return xtensor containing the eigenvalues.
     */
    template <class E, std::enable_if_t<!is_complex<typename E::value_type>::value>* = nullptr>
    auto eigvals(const xexpression<E>& A)
    {
        using value_type = typename E::value_type;

        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> wr(vN);
        xtensor<value_type, 1, layout_type::column_major> wi(vN);

        // TODO check if we can remove allocation and pass nullptr as VL / VR
        std::array<std::size_t, 2> shp({N, N});
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        auto geev_res = lapack::geev(M, 'N', 'N', wr, wi, VL, VR);

        using value_type = typename E::value_type;

        xtensor<std::complex<value_type>, 1> eig_vals;
        eig_vals.reshape({N});

        xt::real(eig_vals) = wr;
        xt::imag(eig_vals) = wi;

        return eig_vals;
    }

    template <class E, std::enable_if_t<is_complex<typename E::value_type>::value>* = nullptr>
    auto eigvals(const xexpression<E>& A)
    {
        using value_type = typename E::value_type;

        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        // TODO check if we can remove allocation and pass nullptr as VL / VR
        std::array<std::size_t, 2> shp({N, N});
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        auto geev_res = lapack::geev(M, 'N', 'N', w, VL, VR);

        using value_type = typename E::value_type;

        return w;
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
        using common_type = std::common_type_t<typename T::value_type, typename O::value_type>;
        using return_type = xarray<common_type>;

        return_type result;

        if (t.dimension() == 1 && o.dimension() == 1)
        {
            result.reshape(std::vector<std::size_t>{1});
            if (is_complex<typename T::value_type>::value)
            {
                blas::dotu(t, o, result(0));
            }
            else
            {
                blas::dot(t, o, result(0));
            }
            return result;
        }
        else
        {
            if (t.dimension() == 2 && o.dimension() == 1)
            {
                result.reshape(std::vector<std::size_t>{t.shape()[0]});
                blas::gemv(t, o, result);
            }
            else if (t.dimension() == 1 && o.dimension() == 2)
            {
                result.reshape(std::vector<std::size_t>{o.shape()[0]});
                blas::gemv(o, t, result, true);
            }
            else if (t.dimension() == 2 && o.dimension() == 2)
            {
                result.reshape(std::vector<std::size_t>{t.shape()[0], o.shape()[1]});
                blas::gemm(o, t, result);
            }
            return result;
        }
        throw std::runtime_error("Dot broadcasting not implemented yet. Only 1- and 2-dim work.");
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
    auto vdot(const xexpression<T>& a, const xexpression<O>& b) {

        using common_type = std::common_type_t<typename T::value_type, typename O::value_type>;

        const auto da = a.derived_cast();
        const auto db = b.derived_cast();

        XTENSOR_ASSERT(da.dimension() == 1);
        XTENSOR_ASSERT(db.dimension() == 1);

        common_type result = 0;
        if (is_complex<typename T::value_type>::value)
        {
            blas::dot(da, db, result);
        }
        else
        {
            blas::dotu(da, db, result);
        }

        return result;
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
    auto outer(const xexpression<T>& a, const xexpression<O>& b) {
        using common_type = std::common_type_t<typename T::value_type, typename O::value_type>;
        using return_type = xtensor<common_type, 2>;

        const auto da = a.derived_cast();
        const auto db = b.derived_cast();

        XTENSOR_ASSERT(da.dimension() == 1);
        XTENSOR_ASSERT(db.dimension() == 1);

        typename return_type::shape_type s = {da.shape()[0], db.shape()[0]};
        return_type result(s, 0);

        blas::ger(da, db, result);

        return result;
    }

    /**
     * Compute the determinant by utilizing LU factorization
     */
    template <class T>
    auto det(const xexpression<T>& A)
    {
        using value_type = typename T::value_type;
        xtensor<value_type, 2, layout_type::column_major> LU = A.derived_cast();

        uvector<XBLAS_INDEX> piv(std::min(LU.shape()[0], LU.shape()[1]));

        int res = lapack::getrf(LU, piv);

        value_type result(1);
        for (std::size_t i = 0; i < piv.size(); ++i)
        {
            if (piv[i] != int(i + 1))
            {
                result *= value_type(-1);
            }
        }

        for (std::size_t i = 0; i < LU.shape()[0]; ++i)
        {
            result *= LU(i, i);
        }
        return result;
    }

    template <class T, std::enable_if_t<is_complex<typename T::value_type>::value, int> = 0>
    auto slogdet(const xexpression<T>& A)
    {
        using value_type = typename T::value_type;
        xtensor<value_type, 2, layout_type::column_major> LU = A.derived_cast();
        uvector<XBLAS_INDEX> piv(std::min(LU.shape()[0], LU.shape()[1]));

        int info = lapack::getrf(LU, piv);

        if (info != 0)
        {
            throw std::runtime_error("LU factorization did not compute.");
        }

        value_type result(0);
        int sign = 0;
        for (std::size_t i = 0; i < piv.size(); ++i)
        {
            sign += (piv[i] != int(i + 1));
        }

        value_type v_sign = (sign % 2) ? -1 : 1;

        for (std::size_t i = 0; i < LU.shape()[0]; ++i)
        {
            auto abs_elem = std::abs(LU(i, i));
            v_sign *= (LU(i, i) / abs_elem);
            result += std::log(abs_elem);
        }
        return std::make_tuple(v_sign, result);
    }

    template <class T, std::enable_if_t<!is_complex<typename T::value_type>::value, int> = 0>
    auto slogdet(const xexpression<T>& A)
    {
        using value_type = typename T::value_type;
        xtensor<value_type, 2, layout_type::column_major> LU = A.derived_cast();
        uvector<XBLAS_INDEX> piv(std::min(LU.shape()[0], LU.shape()[1]));

        int info = lapack::getrf(LU, piv);

        if (info != 0)
        {
            return std::make_tuple(value_type(0),
                                   -std::numeric_limits<value_type>::infinity());
        }

        value_type result(0);
        int sign = 0;

        for (std::size_t i = 0; i < piv.size(); ++i)
        {
            sign += (piv[i] != int(i + 1));
        }

        for (std::size_t i = 0; i < LU.shape()[0]; ++i)
        {
            value_type abs_el = LU(i, i);
            if (abs_el < 0)
            {
                sign += 1;
                abs_el = -abs_el;
            }
            result += std::log(abs_el);
        }

        value_type v_sign = (sign % 2) ? -1 : 1;
        return std::make_tuple(v_sign, result);
    }

    namespace detail
    {
        template <class E>
        inline auto call_gqr(E& A, uvector<typename E::value_type>& tau)
            -> std::enable_if_t<!is_complex<typename E::value_type>::value>
        {
            int info = lapack::orgqr(A, tau);
            if (info > 0)
            {
                throw std::runtime_error("Could not find Q (orgqr).");
            }
        }

        template <class E>
        inline auto call_gqr(E& A, uvector<typename E::value_type>& tau)
            -> std::enable_if_t<is_complex<typename E::value_type>::value>
        {
            int info = lapack::ungqr(A, tau);
            if (info > 0)
            {
                throw std::runtime_error("Could not find Q (ungqr).");
            }
        }
    }
    /**
     * Compute the QR decomposition of \em t.
     * @param t The matrix to calculate Q and R for
     * @return std::tuple with Q and R
     */
    template <class T>
    auto qr(const xexpression<T>& A, bool calculate_q = true)
    {
        using value_type = typename T::value_type;
        using xtype = xtensor<value_type, 2, layout_type::column_major>;

        xtype R = A.derived_cast();
        auto res = lapack::geqrf(R);

        if (std::get<0>(res) != 0)
        {
            throw std::runtime_error("QR decomposition failed.");
        }

        xtype Q;
        if (calculate_q)
        {
            Q = R;
            detail::call_gqr(Q, std::get<1>(res));
        }

        for (std::size_t i = 0; i < R.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < i; j++)
            {
                R(i, j) = 0;
            }
        }

        return std::make_tuple(Q, R);
    }

    /**
     * Compute the Cholesky decomposition of \em t.
     * @return the decomposed matrix
     */
    template <class T>
    auto cholesky(const xexpression<T>& A)
    {
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        int info = lapack::potr(M, 'L');

        if (info > 0)
        {
            throw std::runtime_error("Cholesky decomposition failed.");
        }

        // delete upper triangle
        XTENSOR_ASSERT(M.shape()[0] > 1 && M.shape()[1] > 1);

        for (std::size_t i = 0; i < M.shape()[0]; ++i)
        {
            for (std::size_t j = i + 1; j < M.shape()[1]; ++j)
            {
                M(i, j) = 0;
            }
        }

        return M;
    }

    /**
     * Compute the SVD decomposition of \em A.
     * @return tuple containing S, V, and D
     */
    template <class T>
    auto svd(const xexpression<T>& A, bool compute_uv = true)
    {
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        char job_type = 'A';
        if (!compute_uv)
        {
            job_type = 'N';
        }

        auto result = lapack::gesdd(M, job_type);

        if (std::get<0>(result) > 0)
        {
            throw std::runtime_error("SVD decomposition failed.");
        }

        return std::make_tuple(std::get<1>(result), std::get<2>(result), std::get<3>(result));
    }

    /**
     * Calculate Moore-Rose pseudo inverse using LAPACK SVD.
     */
    template <class T>
    auto pinv(const xexpression<T>& A, double rcond = 1e-15)
    {
        using value_type = typename T::value_type;
        const auto& dA = A.derived_cast();

        xtensor<value_type, 2, layout_type::column_major> M = xt::conj(dA);

        auto gesdd_res = lapack::gesdd(M, 'S');

        if (std::get<0>(gesdd_res) != 0)
        {
            throw std::runtime_error("SVD decomposition failed.");
        }

        auto u = std::get<1>(gesdd_res);
        auto s = std::get<2>(gesdd_res);
        auto vt = std::get<3>(gesdd_res);

        value_type cutoff = rcond * (*std::max_element(s.begin(), s.end()));

        for (std::size_t i = 0; i < s.size(); ++i)
        {
            if (s(i) > cutoff)
            {
                s(i) = 1. / s(i);
            }
            else
            {
                s(i) = 0;
            }
        }
        auto ut = xt::transpose(u);
        auto vww = xt::view(s, xt::all(), xt::newaxis());
        auto m = vww * ut;
        auto vtt = xt::transpose(vt);

        std::array<std::size_t, 2> shp({vtt.shape()[0], m.shape()[1]});
        xtensor<value_type, 2> result(shp);
        blas::gemm(vtt, m, result);
        return result;
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
    auto matrix_power(const xexpression<E>& A, long n)
    {
        using input_type = std::decay_t<E>;
        using value_type = typename input_type::value_type;
        using xtype = xtensor<value_type, 2>;

        // copy input matrix
        xtype mat = A.derived_cast();

        XTENSOR_ASSERT(mat.dimension() == 2);
        XTENSOR_ASSERT(mat.shape()[0] == mat_inp.shape()[1]);

        xtype res(mat.shape());
        if (n == 0)
        {
            res = eye(mat.shape()[0]);
            return res;
        }
        else if (n < 0)
        {
            mat = inv(mat);
            n = -n;
        }


        xtype temp(mat.shape());
        res = mat;
        if (n <= 3)
        {
            for (int i = 0; i < n - 1; ++i)
            {
                blas::gemm(res, mat, temp);
                res = temp;
            }
            return res;
        }

        // if n > 3, do a binary decomposition (copied from NumPy)
        long bits, var = n, i = 0;
        for(bits = 0; var != 0; ++bits)
        {
            var >>= 1;
        }
        while (~n & (1 << i))
        {
            blas::gemm(mat, mat, temp);
            temp = res;
            ++i;
        }
        ++i;
        res = mat;
        for (; i < bits; ++i)
        {
            blas::gemm(mat, mat, temp);
            mat = temp;
            if (n & (1 << i))
            {
                blas::gemm(res, mat, temp);
                res = temp;
            }
        }
        return res;
    }


    /**
     * Compute the trace of a xexpression.
     */
    template <class T>
    auto trace(const xexpression<T>& M, int offset = 0, int axis1 = 0, int axis2 = 1)
    {
        const auto& dM = M.derived_cast();
        auto d = xt::diagonal(dM);

        std::size_t dim = d.dimension();
        if (dim == 1)
        {
            return xt::xarray<double>(xt::sum(d)());
        }
        else
        {
            return xt::xarray<double>(xt::sum(d, {dim - 1}));
        }
    }

    /**
     * Calculate the Kronecker product between two 2D xexpressions.
     */

    template <class T, class E>
    auto kron(const xexpression<T>& a, const xexpression<E>& b)
    {
        using value_type = std::common_type_t<typename T::value_type, typename E::value_type>;

        const auto& da = a.derived_cast();
        const auto& db = b.derived_cast();

        XTENSOR_ASSERT(da.dimension() == 2);
        XTENSOR_ASSERT(db.dimension() == 2);

        std::array<std::size_t, 2> shp({ da.shape()[0] * db.shape()[0], da.shape()[1] * db.shape()[1] });
        xtensor<value_type, 2> res(shp);

        for (std::size_t i = 0; i < da.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < da.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < db.shape()[0]; ++k)
                {
                    for (std::size_t h = 0; h < db.shape()[1]; ++h)
                    {
                        res(i * db.shape()[0] + k, j * db.shape()[1] + h) = da(i, j) * db(k, h);
                    }
                }
            }
        }

        return res;
    }

    /**
     * Calculate the matrix rank of \ref m.
     * If tol == -1, the tolerance is automatically computed.
     *
     * @param m matrix for which rank is calculated
     * @param tol tolerance for finding rank
     */
    template <class T>
    int matrix_rank(const xexpression<T>& m, double tol = -1)
    {
        using value_type = typename T::value_type;
        xtensor<value_type, 2, layout_type::column_major> M = m.derived_cast();

        auto svd_res = svd(m, false);
        auto s = std::get<1>(svd_res);
        auto max_el = std::max_element(s.begin(), s.end());

        if (tol == -1)
        {
            tol = (*max_el) * (double) std::max(M.shape()[0], M.shape()[1]) * std::numeric_limits<value_type>::epsilon();
        }

        int sm = 0;
        for (const auto& el : s)
        {
            if (el > tol)
            {
                ++sm;
            }
        }
        return sm;
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