/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_UTILS_HPP
#define XBLAS_UTILS_HPP

#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "xtensor-blas/xblas_config.hpp"
#include "xflens/cxxblas/typedefs.h"
#include "xtensor/xutils.hpp"

#ifndef DEFAULT_LEADING_STRIDE_BEHAVIOR
#define DEFAULT_LEADING_STRIDE_BEHAVIOR throw std::runtime_error("No valid layout chosen.");
#endif

#ifndef DEFAULT_STORAGE_ORDER_BEHAVIOR
#define DEFAULT_STORAGE_ORDER_BEHAVIOR throw std::runtime_error("Cannot handle layout_type of e.");
#endif

namespace xt
{
    template <layout_type L = layout_type::row_major, class T>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<has_data_interface<std::decay_t<T>>::value && std::decay_t<T>::static_layout == L, T&&>
    {
        return std::forward<T>(t);
    }

    namespace detail
    {
        constexpr layout_type layout_remove_any(const layout_type layout)
        {
            return layout == layout_type::any ? XTENSOR_DEFAULT_LAYOUT : layout;
        }
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<(!has_data_interface<std::decay_t<T>>::value || I::static_layout != L)
                            && detail::is_array<typename I::shape_type>::value,
                            xtensor<typename I::value_type,
                                    std::tuple_size<typename I::shape_type>::value,
                                    detail::layout_remove_any(L)>>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<(!has_data_interface<std::decay_t<T>>::value || I::static_layout != L) &&
                            !detail::is_array<typename I::shape_type>::value,
                            xarray<typename I::value_type, detail::layout_remove_any(L)>>
    {
        return t;
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
        DEFAULT_STORAGE_ORDER_BEHAVIOR;
    }

    /**
     * Get leading stride
     */

    template <class A, std::enable_if_t<A::static_layout == layout_type::row_major>* = nullptr>
    inline BLAS_IDX get_leading_stride(const A& a)
    {
        return static_cast<BLAS_IDX>(a.strides().front() == 0 ? a.shape().back() : a.strides().front());
    }

    template <class A, std::enable_if_t<A::static_layout == layout_type::column_major>* = nullptr>
    inline BLAS_IDX get_leading_stride(const A& a)
    {
        return static_cast<BLAS_IDX>(a.strides().back() == 0 ? a.shape().front() : a.strides().back());
    }

    template <class A, std::enable_if_t<A::static_layout != layout_type::row_major && A::static_layout != layout_type::column_major>* = nullptr>
    inline BLAS_IDX get_leading_stride(const A& a)
    {
        if (a.layout() == layout_type::row_major)
        {
            return static_cast<BLAS_IDX>(a.strides().front() == 0 ? a.shape().back() : a.strides().front());
        }
        else if (a.layout() == layout_type::column_major)
        {
            return static_cast<BLAS_IDX>(a.strides().back() == 0 ? a.shape().front() : a.strides().back());
        }
        DEFAULT_LEADING_STRIDE_BEHAVIOR;
    }

    /********************
     * Get front stride *
     ********************/

    template <class E>
    inline blas_index_t stride_front(const E& e)
    {
        if (E::static_layout == layout_type::column_major)
        {
            return blas_index_t(1);
        }
        else
        {
            return static_cast<blas_index_t>(e.strides().front() == 0 ? 1 : e.strides().front());
        }
    }

    /*******************
     * Get back stride *
     *******************/

    template <class E>
    inline blas_index_t stride_back(const E& e)
    {
        if (E::static_layout == layout_type::row_major)
        {
            return blas_index_t(1);
        }
        else
        {
            return static_cast<blas_index_t>(e.strides().back() == 0 ? 1 : e.strides().back());
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
