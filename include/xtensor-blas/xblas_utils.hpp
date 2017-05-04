/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_UTILS_HPP
#define XBLAS_UTILS_HPP

#include "flens/cxxblas/typedefs.h"
#include "xtensor/xutils.hpp"

namespace xt
{
    template <layout_type L = layout_type::row_major, class T>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<has_raw_data_interface<T>::value && std::decay_t<T>::static_layout == L, T&&>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<(!has_raw_data_interface<T>::value || I::static_layout != L)
                                && detail::is_array<typename I::shape_type>::value,
                            xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value, L>>
    {
        xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value, L> ret = t;
        return ret;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<(!has_raw_data_interface<T>::value || I::static_layout != L) &&
                                !detail::is_array<typename I::shape_type>::value,
                            xarray<typename I::value_type, L>>
    {
        xarray<typename I::value_type, L> ret = t;
        return ret;
    }

    template <layout_type L = layout_type::row_major, class T>
    inline auto copy_to_layout(T&& t)
        -> std::enable_if_t<std::decay_t<T>::static_layout == L, T>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto copy_to_layout(T&& t)
        -> std::enable_if_t<std::decay_t<T>::static_layout != L && detail::is_array<typename I::shape_type>::value,
                            xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value, L>>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto copy_to_layout(T&& t)
        -> std::enable_if_t<std::decay_t<T>::static_layout != L && !detail::is_array<typename I::shape_type>::value,
                            xarray<typename I::value_type, L>>
    {
        return t;
    }

    template <class E>
    inline cxxblas::StorageOrder get_blas_storage_order(const E& e)
    {
        if (e.layout() == layout_type::row_major)
        {
            return cxxblas::StorageOrder::RowMajor;
        }
        else if (e.layout() == layout_type::column_major)
        {
            return cxxblas::StorageOrder::ColMajor;
        }
        throw std::runtime_error("Cannot handle layout_type of e.");
    }

    /**
     * Get leading stride
     */

    template <class A, std::enable_if_t<A::static_layout == layout_type::column_major>* = nullptr>
    inline BLAS_IDX get_leading_stride(const A& a)
    {
        return (BLAS_IDX) a.strides().back();
    }

    template <class A, std::enable_if_t<A::static_layout == layout_type::row_major>* = nullptr>
    inline BLAS_IDX get_leading_stride(const A& a)
    {
        return (BLAS_IDX) a.strides().front();
    }

    template <class A, std::enable_if_t<A::static_layout != layout_type::row_major && A::static_layout != layout_type::column_major>* = nullptr>
    inline BLAS_IDX get_leading_stride(const A& a)
    {
        if (a.layout() == layout_type::row_major)
        {
            return (BLAS_IDX) a.strides().front();
        }
        else if (a.layout() == layout_type::column_major)
        {
            return (BLAS_IDX) a.strides().back();
        }
        else
        {
            throw std::runtime_error("No valid layout chosen.");
        }
    }

    /*******************************
     * is_xfunction implementation *
     *******************************/

    namespace detail
    {
        template<class xF, class xR, class fE, class... xE>
        std::true_type  is_xfunction_impl(const xfunction<xF, xR, fE, xE...>&);

        std::false_type is_xfunction_impl(...);
    }

    template<typename T>
    constexpr bool is_xfunction(T&& t) {
        return decltype(detail::is_xfunction_impl(t))::value;
    }
}
#endif