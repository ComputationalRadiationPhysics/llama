// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Meta.hpp"

#include <array>
#include <type_traits>

namespace llama
{
    /// Represents a coordinate for a record inside the record dimension tree.
    /// \tparam Coords... the compile time coordinate.
    template <std::size_t... Coords>
    struct RecordCoord
    {
        /// The list of integral coordinates as `boost::mp11::mp_list`.
        using List = boost::mp11::mp_list_c<std::size_t, Coords...>;

        static constexpr std::size_t front = boost::mp11::mp_front<List>::value;
        static constexpr std::size_t back = boost::mp11::mp_back<List>::value;
        static constexpr std::size_t size = sizeof...(Coords);
    };

    template <>
    struct RecordCoord<>
    {
        using List = boost::mp11::mp_list_c<std::size_t>;

        static constexpr std::size_t size = 0;
    };

    template <std::size_t... CoordsA, std::size_t... CoordsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<CoordsA...>, RecordCoord<CoordsB...>)
    {
        return false;
    }

    template <std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<Coords...>, RecordCoord<Coords...>)
    {
        return true;
    }

    template <std::size_t... CoordsA, std::size_t... CoordsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(RecordCoord<CoordsA...> a, RecordCoord<CoordsB...> b)
    {
        return !(a == b);
    }

    template <typename T>
    inline constexpr bool isRecordCoord = false;

    template <std::size_t... Coords>
    inline constexpr bool isRecordCoord<RecordCoord<Coords...>> = true;

    inline namespace literals
    {
        /// Literal operator for converting a numeric literal into a \ref RecordCoord.
        template <char... Digits>
        constexpr auto operator"" _RC()
        {
            constexpr auto coord = []() constexpr
            {
                char digits[] = {(Digits - 48)...};
                std::size_t acc = 0;
                std ::size_t powerOf10 = 1;
                for (int i = sizeof...(Digits) - 1; i >= 0; i--)
                {
                    acc += digits[i] * powerOf10;
                    powerOf10 *= 10;
                }
                return acc;
            }
            ();
            return RecordCoord<coord>{};
        }
    } // namespace literals

    namespace internal
    {
        template <class L>
        struct mp_unwrap_sizes_impl;

        template <template <class...> class L, typename... T>
        struct mp_unwrap_sizes_impl<L<T...>>
        {
            using type = RecordCoord<T::value...>;
        };

        template <typename L>
        using mp_unwrap_sizes = typename mp_unwrap_sizes_impl<L>::type;
    } // namespace internal

    /// Converts a type list of integral constants into a \ref RecordCoord.
    template <typename L>
    using RecordCoordFromList = internal::mp_unwrap_sizes<L>;

    /// Concatenate a set of \ref RecordCoord%s.
    template <typename... RecordCoords>
    using Cat = RecordCoordFromList<boost::mp11::mp_append<typename RecordCoords::List...>>;

    /// Concatenate a set of \ref RecordCoord%s instances.
    template <typename... RecordCoords>
    constexpr auto cat(RecordCoords...)
    {
        return Cat<RecordCoords...>{};
    }

    /// RecordCoord without first coordinate component.
    template <typename RecordCoord>
    using PopFront = RecordCoordFromList<boost::mp11::mp_pop_front<typename RecordCoord::List>>;

    namespace internal
    {
        template <typename First, typename Second>
        struct RecordCoordCommonPrefixIsBiggerImpl;

        template <std::size_t... Coords1, std::size_t... Coords2>
        struct RecordCoordCommonPrefixIsBiggerImpl<RecordCoord<Coords1...>, RecordCoord<Coords2...>>
        {
            static constexpr auto value = []() constexpr
            {
                // CTAD does not work if Coords1/2 is an empty pack
                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
                for (auto i = 0; i < std::min(a1.size(), a2.size()); i++)
                {
                    if (a1[i] > a2[i])
                        return true;
                    if (a1[i] < a2[i])
                        return false;
                }
                return false;
            }
            ();
        };
    } // namespace internal

    /// Checks wether the first RecordCoord is bigger than the second.
    template <typename First, typename Second>
    inline constexpr auto RecordCoordCommonPrefixIsBigger
        = internal::RecordCoordCommonPrefixIsBiggerImpl<First, Second>::value;

    namespace internal
    {
        template <typename First, typename Second>
        struct RecordCoordCommonPrefixIsSameImpl;

        template <std::size_t... Coords1, std::size_t... Coords2>
        struct RecordCoordCommonPrefixIsSameImpl<RecordCoord<Coords1...>, RecordCoord<Coords2...>>
        {
            static constexpr auto value = []() constexpr
            {
                // CTAD does not work if Coords1/2 is an empty pack
                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
                for (auto i = 0; i < std::min(a1.size(), a2.size()); i++)
                    if (a1[i] != a2[i])
                        return false;
                return true;
            }
            ();
        };
    } // namespace internal

    /// Checks whether two \ref RecordCoord%s are the same or one is the prefix of the other.
    template <typename First, typename Second>
    inline constexpr auto RecordCoordCommonPrefixIsSame
        = internal::RecordCoordCommonPrefixIsSameImpl<First, Second>::value;
} // namespace llama
