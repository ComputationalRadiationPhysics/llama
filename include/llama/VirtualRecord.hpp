// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "View.hpp"

#include <iosfwd>
#include <type_traits>

namespace llama
{
    template <typename View, typename BoundRecordCoord, bool OwnView>
    struct VirtualRecord;

    template <typename View>
    inline constexpr auto is_VirtualRecord = false;

    template <typename View, typename BoundRecordCoord, bool OwnView>
    inline constexpr auto is_VirtualRecord<VirtualRecord<View, BoundRecordCoord, OwnView>> = true;

    /// A \ref VirtualRecord that owns and holds a single value.
    template <typename RecordDim>
    using One = VirtualRecord<decltype(allocViewStack<0, RecordDim>()), RecordCoord<>, true>;

    /// Creates a single \ref VirtualRecord owning a view with stack memory and copies all values from an existing \ref
    /// VirtualRecord.
    template <typename VirtualRecord>
    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualRecordStack(const VirtualRecord& vd) -> decltype(auto)
    {
        One<typename VirtualRecord::AccessibleRecordDim> temp;
        temp = vd;
        return temp;
    }

    namespace internal
    {
        template <
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(
            LeftRecord& left,
            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> LeftRecord&
        {
            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
            // if the record dimension left and right is the same, a single loop is enough and no tag check is needed.
            // this safes a lot of compilation time.
            if constexpr (std::is_same_v<
                              typename LeftRecord::AccessibleRecordDim,
                              typename RightRecord::AccessibleRecordDim>)
            {
                forEachLeaf<typename LeftRecord::AccessibleRecordDim>([&](auto coord) LLAMA_LAMBDA_INLINE
                                                                      { Functor{}(left(coord), right(coord)); });
            }
            else
            {
                forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                    [&](auto leftCoord) LLAMA_LAMBDA_INLINE
                    {
                        using LeftInnerCoord = decltype(leftCoord);
                        forEachLeaf<typename RightRecord::AccessibleRecordDim>(
                            [&](auto rightCoord) LLAMA_LAMBDA_INLINE
                            {
                                using RightInnerCoord = decltype(rightCoord);
                                if constexpr (hasSameTags<
                                                  typename LeftRecord::AccessibleRecordDim,
                                                  LeftInnerCoord,
                                                  typename RightRecord::AccessibleRecordDim,
                                                  RightInnerCoord>)
                                {
                                    Functor{}(left(leftCoord), right(rightCoord));
                                }
                            });
                    });
            }
            return left;
        }

        template <typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(LeftRecord& left, const T& right) -> LeftRecord&
        {
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>([&](auto leftCoord) LLAMA_LAMBDA_INLINE
                                                                  { Functor{}(left(leftCoord), right); });
            return left;
        }

        template <
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(
            const LeftRecord& left,
            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> bool
        {
            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
            bool result = true;
            // if the record dimension left and right is the same, a single loop is enough and no tag check is needed.
            // this safes a lot of compilation time.
            if constexpr (std::is_same_v<
                              typename LeftRecord::AccessibleRecordDim,
                              typename RightRecord::AccessibleRecordDim>)
            {
                forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                    [&](auto coord) LLAMA_LAMBDA_INLINE { result &= Functor{}(left(coord), right(coord)); });
            }
            else
            {
                forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                    [&](auto leftCoord) LLAMA_LAMBDA_INLINE
                    {
                        using LeftInnerCoord = decltype(leftCoord);
                        forEachLeaf<typename RightRecord::AccessibleRecordDim>(
                            [&](auto rightCoord) LLAMA_LAMBDA_INLINE
                            {
                                using RightInnerCoord = decltype(rightCoord);
                                if constexpr (hasSameTags<
                                                  typename LeftRecord::AccessibleRecordDim,
                                                  LeftInnerCoord,
                                                  typename RightRecord::AccessibleRecordDim,
                                                  RightInnerCoord>)
                                {
                                    result &= Functor{}(left(leftCoord), right(rightCoord));
                                }
                            });
                    });
            }
            return result;
        }

        template <typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(const LeftRecord& left, const T& right) -> bool
        {
            bool result = true;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord) LLAMA_LAMBDA_INLINE {
                    result &= Functor{}(
                        left(leftCoord),
                        static_cast<std::remove_reference_t<decltype(left(leftCoord))>>(right));
                });
            return result;
        }

        struct Assign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a = b;
            }
        };

        struct PlusAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a += b;
            }
        };

        struct MinusAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a -= b;
            }
        };

        struct MultiplyAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a *= b;
            }
        };

        struct DivideAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a /= b;
            }
        };

        struct ModuloAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a %= b;
            }
        };

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(TWithOptionalConst& leaf, T) -> std::
            enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::reference_wrapper<TWithOptionalConst>>
        {
            return leaf;
        }

        template <typename VirtualRecord, typename T, std::size_t N, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImplArr(VirtualRecord&& vd, T(&&a)[N], std::index_sequence<Is...>)
        {
            return std::make_tuple(asTupleImpl(vd(RecordCoord<Is>{}), T{})...);
        }

        template <typename VirtualRecord, typename T, std::size_t N>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(VirtualRecord&& vd, T(&&a)[N])
        {
            return asTupleImplArr(std::move(vd), std::move(a), std::make_index_sequence<N>{});
        }

        template <typename VirtualRecord, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(VirtualRecord&& vd, Record<Fields...>)
        {
            return std::make_tuple(asTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T)
            -> std::enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::tuple<TWithOptionalConst&>>
        {
            return {leaf};
        }

        template <typename VirtualRecord, typename T, std::size_t N, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImplArr(VirtualRecord&& vd, T(&&a)[N], std::index_sequence<Is...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(RecordCoord<Is>{}), T{})...);
        }

        template <typename VirtualRecord, typename T, std::size_t N>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(VirtualRecord&& vd, T(&&a)[N])
        {
            return asFlatTupleImplArr(std::move(vd), std::move(a), std::make_index_sequence<N>{});
        }

        template <typename VirtualRecord, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(VirtualRecord&& vd, Record<Fields...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template <typename T, typename = void>
        constexpr inline auto isTupleLike = false;

        // get<I>(t) and std::tuple_size<T> must be available
        using std::get; // make sure a get<0>() can be found, so the compiler can compile the trait
        template <typename T>
        constexpr inline auto
            isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>> = true;

        template <typename... Ts>
        constexpr inline auto dependentFalse = false;

        template <typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>);

        template <typename T1, typename T2>
        LLAMA_FN_HOST_ACC_INLINE void assignTupleElement(T1&& dst, T2&& src)
        {
            if constexpr (isTupleLike<std::decay_t<T1>> && isTupleLike<std::decay_t<T2>>)
            {
                static_assert(std::tuple_size_v<std::decay_t<T1>> == std::tuple_size_v<std::decay_t<T2>>);
                assignTuples(dst, src, std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>{});
            }
            else if constexpr (!isTupleLike<std::decay_t<T1>> && !isTupleLike<std::decay_t<T2>>)
                std::forward<T1>(dst) = std::forward<T2>(src);
            else
                static_assert(dependentFalse<T1, T2>, "Elements to assign are not tuple/tuple or non-tuple/non-tuple.");
        }

        template <typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>)
        {
            static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>);
            using std::get;
            (assignTupleElement(get<Is>(std::forward<Tuple1>(dst)), get<Is>(std::forward<Tuple2>(src))), ...);
        }

        template <typename T, typename Tuple, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeFromTuple(Tuple&& src, std::index_sequence<Is...>)
        {
            using std::get;
            return T{get<Is>(std::forward<Tuple>(src))...};
        }

        template <typename T, typename SFINAE, typename... Args>
        constexpr inline auto isDirectListInitializableImpl = false;

        template <typename T, typename... Args>
        constexpr inline auto
            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...> = true;

        template <typename T, typename... Args>
        constexpr inline auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

        template <typename T, typename Tuple>
        constexpr inline auto isDirectListInitializableFromTuple = false;

        template <typename T, template <typename...> typename Tuple, typename... Args>
        constexpr inline auto
            isDirectListInitializableFromTuple<T, Tuple<Args...>> = isDirectListInitializable<T, Args...>;

        template <typename RecordDim, typename RecordCoord>
        constexpr inline auto unboundArraysUntil = []() constexpr
        {
            std::size_t count = 0;
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<RecordCoord::size>>(
                [&](auto i) constexpr
                {
                    using RC = RecordCoordFromList<boost::mp11::mp_take_c<typename RecordCoord::List, i>>;
                    using TypeAtRC = GetType<RecordDim, RC>;
                    count += static_cast<std::size_t>(internal::is_unbounded_array_v<TypeAtRC>);
                });
            return count;
        }
        ();
    } // namespace internal

    /// Virtual record type returned by \ref View after resolving an array dimensions coordinate or partially resolving
    /// a \ref RecordCoord. A virtual record does not hold data itself (thus named "virtual"), it just binds enough
    /// information (array dimensions coord and partial record coord) to retrieve it from a \ref View later. Virtual
    /// records should not be created by the user. They are returned from various access functions in \ref View and
    /// VirtualRecord itself.
    template <typename T_View, typename BoundRecordCoord, bool OwnView>
    struct VirtualRecord
    {
        using View = T_View; ///< View this virtual record points into.

    private:
        using ArrayDims = typename View::Mapping::ArrayDims;
        using RecordDim = typename View::Mapping::RecordDim;
        using DynamicArrayExtentsArray = Array<std::size_t, internal::unboundArraysUntil<RecordDim, BoundRecordCoord>>;

        std::conditional_t<OwnView, View, View&> view;
        [[no_unique_address]] const ArrayDims arrayDimsCoord;
        [[no_unique_address]] const DynamicArrayExtentsArray dynamicArrayExtents;

    public:
        /// Subtree of the record dimension of View starting at BoundRecordCoord. If BoundRecordCoord is `RecordCoord<>`
        /// (default) AccessibleRecordDim is the same as `Mapping::RecordDim`.
        using AccessibleRecordDim = GetType<RecordDim, BoundRecordCoord>;

        /// Creates an empty VirtualRecord. Only available for if the view is owned. Used by llama::One.
        LLAMA_FN_HOST_ACC_INLINE VirtualRecord()
            /* requires(OwnView) */
            : arrayDimsCoord({})
            , dynamicArrayExtents({})
            , view{allocViewStack<0, RecordDim>()}
        {
            static_assert(OwnView, "The default constructor of VirtualRecord is only available if it owns the view.");
        }

        LLAMA_FN_HOST_ACC_INLINE
        VirtualRecord(
            std::conditional_t<OwnView, View&&, View&> view,
            ArrayDims arrayDimsCoord,
            DynamicArrayExtentsArray dynamicArrayExtents = {})
            : view{static_cast<decltype(view)>(view)}
            , arrayDimsCoord(arrayDimsCoord)
            , dynamicArrayExtents{dynamicArrayExtents}
        {
        }

        VirtualRecord(const VirtualRecord&) = default;
        VirtualRecord(VirtualRecord&&) = default;

        /// Create a VirtuaRecord from a different VirtualRecord. Only available for if the view is owned. Used by
        /// llama::One.
        template <typename OtherView, typename OtherBoundRecordCoord, bool OtherOwnView>
        LLAMA_FN_HOST_ACC_INLINE VirtualRecord(
            const VirtualRecord<OtherView, OtherBoundRecordCoord, OtherOwnView>& virtualRecord)
            /* requires(OwnView) */
            : VirtualRecord()
        {
            static_assert(
                OwnView,
                "The copy constructor of VirtualRecord from a different VirtualRecord is only available if it owns the "
                "view.");
            *this = virtualRecord;
        }

        // TODO: unify with previous in C++20 and use explicit(cond)
        /// Create a VirtuaRecord from a scalar. Only available for if the view is owned. Used by llama::One.
        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE explicit VirtualRecord(const T& scalar)
            /* requires(OwnView) */
            : VirtualRecord()
        {
            static_assert(
                OwnView,
                "The copy constructor of VirtualRecord from a scalar is only available if it owns the view.");
            *this = scalar;
        }

        /// Access a record in the record dimension underneath the current virtual record using a \ref RecordCoord. If
        /// the access resolves to a leaf, a reference to a variable inside the \ref View storage is returned, otherwise
        /// another virtual record.
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            using AccessedType = GetType<RecordDim, AbsolutCoord>;
            if constexpr (
                isRecord<AccessedType> || internal::is_bounded_array<AccessedType>::value
                || internal::is_unbounded_array_v<AccessedType>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<const View, AbsolutCoord>{
                    this->view,
                    arrayDimsCoord,
                    dynamicArrayExtents,
                };
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayDimsCoord, dynamicArrayExtents, AbsolutCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> coord = {}) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            using AccessedType = GetType<RecordDim, AbsolutCoord>;
            if constexpr (
                isRecord<AccessedType> || internal::is_bounded_array<AccessedType>::value
                || internal::is_unbounded_array_v<AccessedType>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<View, AbsolutCoord>{this->view, arrayDimsCoord, dynamicArrayExtents};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayDimsCoord, dynamicArrayExtents, AbsolutCoord{});
            }
        }

        /// Access a record in the record dimension underneath the current virtual record using a series of tags. If the
        /// access resolves to a leaf, a reference to a variable inside the \ref View storage is returned, otherwise
        /// another virtual record.
        template <typename... Tags, std::enable_if_t<!std::disjunction_v<std::is_integral<Tags>...>, bool> = true>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template <typename... Tags, std::enable_if_t<!std::disjunction_v<std::is_integral<Tags>...>, bool> = true>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        template <
            typename ADD = AccessibleRecordDim,
            std::enable_if_t<internal::is_unbounded_array_v<ADD>, bool> = true>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t i) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<dynamic>>;
            using ResolvedType = GetType<RecordDim, AbsolutCoord>;
            auto newDynamicArrayExtents = push_back(dynamicArrayExtents, i);
            if constexpr (isRecord<ResolvedType> || internal::is_unbounded_array_v<ResolvedType>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<const View, AbsolutCoord>{this->view, arrayDimsCoord, newDynamicArrayExtents};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayDimsCoord, newDynamicArrayExtents, AbsolutCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template <
            typename ADD = AccessibleRecordDim,
            std::enable_if_t<internal::is_unbounded_array_v<ADD>, bool> = true>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t i) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<dynamic>>;
            using ResolvedType = GetType<RecordDim, AbsolutCoord>;
            auto newDynamicArrayExtents = push_back(dynamicArrayExtents, i);
            if constexpr (isRecord<ResolvedType> || internal::is_unbounded_array_v<ResolvedType>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<View, AbsolutCoord>{this->view, arrayDimsCoord, newDynamicArrayExtents};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayDimsCoord, newDynamicArrayExtents, AbsolutCoord{});
            }
        }

        // we need this one to disable the compiler generated copy assignment
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualRecord& other) -> VirtualRecord&
        {
            return this->operator=<VirtualRecord>(other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::Assign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::PlusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::MinusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::MultiplyAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::DivideAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::ModuloAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) += t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const VirtualRecord& vd)
        {
            return vd + t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) -= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) *= t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const VirtualRecord& vd)
        {
            return vd * t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) /= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) %= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd == t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::not_equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd != t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::less<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd > t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::less_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd >= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::greater<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd < t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::greater_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd <= t;
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple()
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple() const
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple()
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple() const
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() const -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        struct Loader
        {
            VirtualRecord& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T()
            {
                return vd.loadAs<T>();
            }
        };

        struct LoaderConst
        {
            const VirtualRecord& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T() const
            {
                return vd.loadAs<T>();
            }
        };

        LLAMA_FN_HOST_ACC_INLINE auto load() -> Loader
        {
            return {*this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto load() const -> LoaderConst
        {
            return {*this};
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE void store(const TupleLike& t)
        {
            internal::assignTuples(asTuple(), t, std::make_index_sequence<std::tuple_size_v<TupleLike>>{});
        }
    };

    template <typename View, typename BoundRecordCoord, bool OwnView>
    auto operator<<(std::ostream& os, const VirtualRecord<View, BoundRecordCoord, OwnView>& vr) -> std::ostream&
    {
        using RecordDim = typename VirtualRecord<View, BoundRecordCoord, OwnView>::AccessibleRecordDim;
        os << "{";
        // TODO: I tried refactoring both branches into one, but MSVC and icpc have troubles with correctly discarding
        // the discarded if constexpr branch and not instantiating templates inside them.
        if constexpr (std::is_array_v<RecordDim>)
        {
            constexpr auto size = std::extent_v<RecordDim>;
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<size>>(
                [&](auto ic)
                {
                    constexpr std::size_t i = decltype(ic)::value;
                    os << '[' << i << ']' << ": " << vr(RecordCoord<i>{});
                    if (i + 1 < size)
                        os << ", ";
                });
        }
        else
        {
            constexpr auto size = boost::mp11::mp_size<RecordDim>::value;
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<size>>(
                [&](auto ic)
                {
                    constexpr std::size_t i = decltype(ic)::value;
                    using Field = boost::mp11::mp_at_c<RecordDim, i>;
                    using Tag = GetFieldTag<Field>;
                    os << structName<Tag>() << ": " << vr(RecordCoord<i>{});
                    if (i + 1 < size)
                        os << ", ";
                });
        }
        os << "}";
        return os;
    }
} // namespace llama

template <typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_size<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
    : boost::mp11::mp_size<typename llama::VirtualRecord<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
{
};

template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
{
    using type = decltype(std::declval<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
};

template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
{
    using type
        = decltype(std::declval<const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
};
