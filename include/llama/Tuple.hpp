// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Meta.hpp"
#include "macros.hpp"

namespace llama
{
    template <typename... Elements>
    struct Tuple
    {
    };

    /// Tuple class like `std::tuple` but suitable for use with offloading devices like GPUs.
    template <typename T_FirstElement, typename... Elements>
    struct Tuple<T_FirstElement, Elements...>
    {
        using FirstElement = T_FirstElement;
        using RestTuple = Tuple<Elements...>;

        constexpr Tuple() = default;

        /// Construct a tuple from values of the same types as the tuple stores.
        LLAMA_FN_HOST_ACC_INLINE constexpr Tuple(FirstElement first, Elements... rest)
            : first(std::move(first))
            , rest(std::move(rest)...)
        {
        }

        // icpc fails to compile the treemap tests with this ctor
#ifndef __INTEL_COMPILER
        /// Construct a tuple from forwarded values of potentially different types as the tuple stores.
        // SFINAE away this ctor if tuple elements cannot be constructed from ctor arguments
        template <
            typename T,
            typename... Ts,
            std::enable_if_t<
                sizeof...(Elements) == sizeof...(Ts)
                    && std::is_constructible_v<T_FirstElement, T> && (std::is_constructible_v<Elements, Ts> && ...),
                int> = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr Tuple(T&& firstArg, Ts&&... restArgs)
            : first(std::forward<T>(firstArg))
            , rest(std::forward<Ts>(restArgs)...)
        {
        }
#endif

        FirstElement first; ///< the first element (if existing)
#ifndef __NVCC__
        [[no_unique_address]] // nvcc 11.3 ICE
#endif
        RestTuple rest; ///< the remaining elements
    };

    template <typename... Elements>
    Tuple(Elements...) -> Tuple<std::remove_cv_t<std::remove_reference_t<Elements>>...>;

    template <std::size_t Pos, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto get(Tuple<Elements...>& tuple) -> auto&
    {
        if constexpr (Pos == 0)
            return tuple.first;
        else
            return get<Pos - 1>(tuple.rest);
    }

    template <std::size_t Pos, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto get(const Tuple<Elements...>& tuple) -> const auto&
    {
        if constexpr (Pos == 0)
            return tuple.first;
        else
            return get<Pos - 1>(tuple.rest);
    }
} // namespace llama

template <typename... Elements>
struct std::tuple_size<llama::Tuple<Elements...>>
{
    static constexpr auto value = sizeof...(Elements);
};

template <std::size_t I, typename... Elements>
struct std::tuple_element<I, llama::Tuple<Elements...>>
{
    using type = boost::mp11::mp_at_c<llama::Tuple<Elements...>, I>;
};

namespace llama
{
    namespace internal
    {
        template <typename... Elements, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto areEqual(
            const Tuple<Elements...>& a,
            const Tuple<Elements...>& b,
            std::index_sequence<Is...>) -> bool
        {
            return ((get<Is>(a) == get<Is>(b)) && ...);
        }
    } // namespace internal

    template <typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        using namespace boost::mp11;
        if constexpr (sizeof...(ElementsA) == sizeof...(ElementsB))
            if constexpr (mp_apply<mp_all, mp_transform<std::is_same, mp_list<ElementsA...>, mp_list<ElementsB...>>>::
                              value)
                return internal::areEqual(a, b, std::make_index_sequence<sizeof...(ElementsA)>{});
        return false;
    }

    template <typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        return !(a == b);
    }

    namespace internal
    {
        template <typename Tuple1, typename Tuple2, size_t... Is1, size_t... Is2>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleCatImpl(
            const Tuple1& t1,
            const Tuple2& t2,
            std::index_sequence<Is1...>,
            std::index_sequence<Is2...>)
        {
            return Tuple{get<Is1>(t1)..., get<Is2>(t2)...};
        }
    } // namespace internal

    template <typename Tuple1, typename Tuple2>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleCat(const Tuple1& t1, const Tuple2& t2)
    {
        return internal::tupleCatImpl(
            t1,
            t2,
            std::make_index_sequence<std::tuple_size_v<Tuple1>>{},
            std::make_index_sequence<std::tuple_size_v<Tuple2>>{});
    }

    namespace internal
    {
        template <std::size_t Pos, typename Tuple, typename Replacement>
        struct TupleReplaceImpl
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(Tuple const tuple, Replacement const replacement)
            {
                return tupleCat(
                    llama::Tuple{tuple.first},
                    TupleReplaceImpl<Pos - 1, typename Tuple::RestTuple, Replacement>()(tuple.rest, replacement));
            };
        };

        template <typename... Elements, typename Replacement>
        struct TupleReplaceImpl<0, Tuple<Elements...>, Replacement>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(Tuple<Elements...> tuple, Replacement const replacement)
            {
                return tupleCat(Tuple{replacement}, tuple.rest);
            };
        };

        template <typename OneElement, typename Replacement>
        struct TupleReplaceImpl<0, Tuple<OneElement>, Replacement>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(Tuple<OneElement>, Replacement const replacement)
            {
                return Tuple{replacement};
            }
        };
    } // namespace internal

    /// Creates a copy of a tuple with the element at position Pos replaced by replacement.
    template <std::size_t Pos, typename Tuple, typename Replacement>
    LLAMA_FN_HOST_ACC_INLINE auto tupleReplace(Tuple tuple, Replacement replacement)
    {
        return internal::TupleReplaceImpl<Pos, Tuple, Replacement>()(tuple, replacement);
    }

    namespace internal
    {
        template <size_t... Is, typename... Elements, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransformHelper(
            std::index_sequence<Is...>,
            const Tuple<Elements...>& tuple,
            const Functor& functor)
        {
            // FIXME(bgruber): nvcc fails to compile
            // Tuple{functor(get<Is>(tuple))...}
            return Tuple<decltype(functor(std::declval<Elements>()))...>{functor(get<Is>(tuple))...};
        }
    } // namespace internal

    /// Applies a functor to every element of a tuple, creating a new tuple with the result of the element
    /// transformations. The functor needs to implement a template `operator()` to which all tuple elements are passed.
    // TODO: replace by mp11 version in Boost 1.74.
    template <typename... Elements, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransform(const Tuple<Elements...>& tuple, const Functor& functor)
    {
        return internal::tupleTransformHelper(std::make_index_sequence<sizeof...(Elements)>{}, tuple, functor);
    }

    /// Returns a copy of the tuple without the first element.
    template <typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_front(const Tuple<Elements...>& tuple)
    {
        return tuple.rest;
    }
} // namespace llama
