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

namespace xt
{
    template <typename T>
    class has_raw_data_interface
    {
        template <typename C>
        static std::true_type test(decltype(std::declval<C>().raw_data_offset()));

        template <typename C>
        static std::false_type test(...);

    public:
        constexpr static bool value = decltype(test<T>(std::size_t(0)))::value == true;
    };


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
        return xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value, L>(std::forward<T>(t));
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
        cxxblas::StorageOrder storage_order;
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

    namespace detail
    {
        template <class A, class B, class T>
        struct get_type_impl {
            using type = xarray<T>;
        };

        template <class A, std::size_t N, class B, std::size_t M, class T>
        struct get_type_impl<std::array<A, N>, std::array<B, M>, T> {
            using type = xtensor<T, M>;
        };

    }

    template <class A, class B>
    struct select_xtype
    {
        using type = typename std::remove_const_t<
                                typename detail::get_type_impl<typename A::shape_type, typename B::shape_type, 
                                                               typename std::common_type_t<typename A::value_type, 
                                typename B::value_type>>::type>;
    };
}

#endif