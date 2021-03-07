// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Meta.hpp"
#include "../Tuple.hpp"
#include "AoS.hpp"
#include "Common.hpp"

namespace llama
{
    using EndOffsetType = std::size_t;
    using SizeType = std::size_t;

    template <typename Tag>
    struct EndOffset
    {
    };
    template <typename Tag>
    struct Size
    {
    };
} // namespace llama

namespace llama::mapping
{
    namespace internal
    {
        using namespace boost::mp11;

        template <typename T>
        inline constexpr bool isEndOffsetField = false;

        template <typename Tag>
        inline constexpr bool isEndOffsetField<EndOffset<Tag>> = true;

        template <typename T>
        inline constexpr bool isSizeField = false;

        template <typename Tag>
        inline constexpr bool isSizeField<Size<Tag>> = true;

        template <typename Field>
        struct AddOffsetAndSizeFieldsImpl
        {
            using type = Record<Field>;
        };

        template <typename Tag, typename Type>
        struct AddOffsetAndSizeFieldsImpl<Field<Tag, Type[]>>
        {
            using type = Record<Field<Tag, Type[]>, Field<EndOffset<Tag>, EndOffsetType>, Field<Size<Tag>, SizeType>>;
        };

        template <typename Field>
        using AddOffsetAndSizeFields = typename AddOffsetAndSizeFieldsImpl<Field>::type;

        template <typename T, typename RecordCoord>
        struct ReplaceDynamicSubarrays
        {
            using Replaced = T;
            using SubRecordDims = mp_list<>;
            using SplitCoords = mp_list<>;
            using Augmented = T;
        };

        template <typename T, std::size_t... RC>
        struct ReplaceDynamicSubarrays<T[], RecordCoord<RC...>>
        {
            using Replaced = EndOffsetType; // offset table entry
            using SubRecordDims = mp_list<typename ReplaceDynamicSubarrays<T, RecordCoord<RC..., dynamic>>::Replaced>;
            using SplitCoords = mp_push_front<
                typename ReplaceDynamicSubarrays<T, RecordCoord<RC..., dynamic>>::SplitCoords,
                RecordCoord<RC...>>;
            using Augmented = T[];
        };

        template <typename Rec, typename RC, typename IS>
        struct ReplaceDynamicSubarraysHelp;

        template <typename Rec, std::size_t... RC, std::size_t... Is>
        struct ReplaceDynamicSubarraysHelp<Rec, RecordCoord<RC...>, std::index_sequence<Is...>>
        {
            using Replaced = Record<Field<
                GetFieldTag<mp_at_c<Rec, Is>>,
                typename ReplaceDynamicSubarrays<GetFieldType<mp_at_c<Rec, Is>>, RecordCoord<RC..., Is>>::Replaced>...>;
            using SubRecordDims
                = mp_append<typename ReplaceDynamicSubarrays<GetFieldType<mp_at_c<Rec, Is>>, RecordCoord<RC..., Is>>::
                                SubRecordDims...>;
            using SplitCoords
                = mp_append<typename ReplaceDynamicSubarrays<GetFieldType<mp_at_c<Rec, Is>>, RecordCoord<RC..., Is>>::
                                SplitCoords...>;

            using Augmented = mp_flatten<mp_transform<AddOffsetAndSizeFields, Rec>>;
        };

        template <typename... Fields, std::size_t... RC>
        struct ReplaceDynamicSubarrays<Record<Fields...>, RecordCoord<RC...>>
            : ReplaceDynamicSubarraysHelp<
                  Record<Fields...>,
                  RecordCoord<RC...>,
                  std::make_index_sequence<sizeof...(Fields)>>
        {
        };

        template <typename RC>
        using BeforeDynamic
            = RecordCoordFromList<mp_take<typename RC::List, mp_find<typename RC::List, mp_size_t<dynamic>>>>;

        template <typename RC>
        using AfterDynamic = RecordCoordFromList<mp_drop<
            typename RC::List,
            mp_size_t<std::min(mp_find<typename RC::List, mp_size_t<dynamic>>::value + 1, RC::size)>>>;

        template <typename RC, std::ptrdiff_t Offset>
        using OffsetLastCoord = RecordCoordFromList<
            mp_push_back<mp_take_c<typename RC::List, RC::size - 1>, mp_size_t<RC::back + Offset>>>;

        template <typename RecordDim, typename RecordCoord>
        struct ShiftRecordCoord;

        template <typename RecordDim>
        struct ShiftRecordCoord<RecordDim, RecordCoord<>>
        {
            using Coord = RecordCoord<>;
        };

        template <typename RecordDim, std::size_t First, std::size_t... Rest>
        struct ShiftRecordCoord<RecordDim, RecordCoord<First, Rest...>>
        {
            template <typename Field>
            using IsUnboundArrayField = llama::internal::is_unbounded_array<GetFieldType<Field>>;

            using ShiftedFirst
                = RecordCoord<First - 2 * mp_count_if<mp_take_c<RecordDim, First>, IsUnboundArrayField>::value>;
            using ShiftedRest = typename ShiftRecordCoord<mp_at_c<RecordDim, First>, RecordCoord<Rest...>>::Coord;

            using Coord = Cat<ShiftedFirst, ShiftedRest>;
        };
    } // namespace internal

    /// A type list containing mappings.
    template <template <typename, typename> typename... SubMappings>
    struct MappingList;

    namespace internal
    {
        template <typename SubRecordDims, typename Mappings>
        struct MapSubRecordDims;

        template <typename... SubRecordDims, template <typename, typename> typename... SubMappings>
        struct MapSubRecordDims<boost::mp11::mp_list<SubRecordDims...>, MappingList<SubMappings...>>
        {
            static_assert(
                sizeof...(SubRecordDims) == sizeof...(SubMappings),
                "There must be as many mappings as sub record dimensions");
            using List = boost::mp11::mp_list<SubMappings<ArrayDims<1>, SubRecordDims>...>;
        };

        template <typename... SubRecordDims, template <typename, typename> typename Mapping>
        struct MapSubRecordDims<boost::mp11::mp_list<SubRecordDims...>, MappingList<Mapping>>
        {
        private:
            template <typename SubRecordDim>
            using MapRecordDim = Mapping<ArrayDims<1>, SubRecordDim>;

        public:
            using List = boost::mp11::mp_transform<MapRecordDim, boost::mp11::mp_list<SubRecordDims...>>;
        };
    } // namespace internal

    /// Meta mapping splitting off sub branches of the given record dimension tree at each field which's type is a
    /// dynamic array. Each dynamic array field is then replaced by an integral offset of type \ref EndOffsetType. This
    /// offset is used to navigate from a virtual record into a dynamic sub array member using a dynamic index. Two
    /// computed fields are added per dynamic array field, which are named \ref EndOffset and \ref Size, giving access
    /// to the offset value and the size of a dynamic sub array. The list of sub record dimensions is then further
    /// mapped using a list of sub mappings.
    ///
    /// @tparam T_RecordDim A record dimension, possibly including field types which are dynamic arrays.
    /// @tparam SubMappings A \ref MappingList of mappings that will be used to map the sub record dimensions after
    /// splitting T_RecordDim at each dynamic array field. If the mapping list contains a single mapping, this one will
    /// be used to map all sub record dimensions. Otherwise, a mapping needs to be given for each sub record dimension.
    template <typename T_ArrayDims, typename T_RecordDim, typename SubMappings = MappingList<PreconfiguredAoS<>::type>>
    struct OffsetTable
    {
        using RDS = internal::ReplaceDynamicSubarrays<T_RecordDim, RecordCoord<>>;
        using SubRecordDims = boost::mp11::mp_push_front<typename RDS::SubRecordDims, typename RDS::Replaced>;
        using SplitCoords = typename RDS::SplitCoords;

        using MappedSubRecordDims = typename internal::MapSubRecordDims<SubRecordDims, SubMappings>::List;

        boost::mp11::mp_rename<MappedSubRecordDims, Tuple> subMappings;

        using ArrayDims = T_ArrayDims;
        using RecordDim = typename RDS::Augmented;
        static constexpr std::size_t blobCount = []() constexpr
        {
            std::size_t count = 0;
            boost::mp11::mp_for_each<boost::mp11::mp_transform<boost::mp11::mp_identity, MappedSubRecordDims>>(
                [&](auto subMapping) { count += decltype(subMapping)::type::blobCount; });
            return count;
        }
        ();

        constexpr OffsetTable() = default;

        template <typename... ArrayDims>
        LLAMA_FN_HOST_ACC_INLINE constexpr OffsetTable(ArrayDims... sizes) : subMappings(sizes...)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return get<0>(subMappings).arrayDims();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
        {
            std::size_t result = 0;
            boost::mp11::mp_for_each<boost::mp11::mp_iota<boost::mp11::mp_size<MappedSubRecordDims>>>(
                [&](auto jc)
                {
                    constexpr auto j = decltype(jc)::value;
                    constexpr auto subBlobs = boost::mp11::mp_at_c<MappedSubRecordDims, j>::blobCount;
                    if (i < subBlobs)
                        result = get<j>(subMappings).blobSize(i);
                    i -= subBlobs;
                });
            return result;
        }

        template <std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template <std::size_t N, typename RecordCoord, typename Blob>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents,
            RecordCoord rc,
            Array<Blob, blobCount>& blobs) const -> decltype(auto)
        {
            return computeRecursive<0>(llama::RecordCoord{}, rc, coord, dynamicArrayExtents, blobs);
        }

    private:
        template <
            std::size_t MappingIndex,
            typename ResolvedRecordCoord,
            typename UnresolvedRecordCoord,
            std::size_t N,
            typename Blob>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto computeRecursive(
            ResolvedRecordCoord,
            UnresolvedRecordCoord,
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents,
            Array<Blob, blobCount>& blobs) const -> decltype(auto)
        {
            static_assert(
                ArrayDims::rank == 1,
                "Not implemented"); // this would need a way to get the prev of coord, also ArrayDims can be a different
                                    // type during recursive instantiation

            using UnresolvedBeforeDynamic = internal::BeforeDynamic<UnresolvedRecordCoord>;
            using UnresolvedAfterDynamic = internal::AfterDynamic<UnresolvedRecordCoord>;
            using ResolvedSoFar = Cat<ResolvedRecordCoord, UnresolvedBeforeDynamic>;

            auto loadBeginOffset = [&](auto unresolvedBeforeDynamic) -> EndOffsetType
            {
                if (coord == ArrayDims{}) [[unlikely]]
                    return 0;
                auto prevCoord = coord;
                prevCoord[0]--;
                return reinterpret_cast<const EndOffsetType&>(
                    *mapToAddress<MappingIndex>(ResolvedRecordCoord{}, unresolvedBeforeDynamic, prevCoord, blobs));
            };

            using Tag = GetTag<RecordDim, ResolvedSoFar>;
            if constexpr (internal::isEndOffsetField<Tag>)
                // load offset from dynamic array member field at prev record coord
                return reinterpret_cast<EndOffsetType&>(*mapToAddress<MappingIndex>(
                    ResolvedRecordCoord{},
                    internal::OffsetLastCoord<UnresolvedBeforeDynamic, -1>{},
                    coord,
                    blobs));
            else if constexpr (internal::isSizeField<Tag>)
            {
                // compute size from end offset and prev end offset (or 0 for the first sub array)
                const auto begin = loadBeginOffset(internal::OffsetLastCoord<UnresolvedBeforeDynamic, -2>{});
                const auto end = reinterpret_cast<const EndOffsetType&>(*mapToAddress<MappingIndex>(
                    ResolvedRecordCoord{},
                    internal::OffsetLastCoord<UnresolvedBeforeDynamic, -2>{},
                    coord,
                    blobs));
                return static_cast<SizeType>(end - begin);
            }
            else if constexpr (std::is_same_v<UnresolvedBeforeDynamic, UnresolvedRecordCoord>)
            {
                // no dynamic sub arrays anymore, proceed with access
                static_assert(N == 0);
                using Type = GetType<RecordDim, ResolvedSoFar>;
                return reinterpret_cast<Type&>(
                    *mapToAddress<MappingIndex>(ResolvedRecordCoord{}, UnresolvedBeforeDynamic{}, coord, blobs));
            }
            else
            {
                // continue resolving with next submapping
                using ShiftedCoord = typename internal::ShiftRecordCoord<RecordDim, ResolvedSoFar>::Coord;
                constexpr auto nextSubMappingIndex = boost::mp11::mp_find<SplitCoords, ShiftedCoord>::value + 1;
                static_assert(nextSubMappingIndex < boost::mp11::mp_size<MappedSubRecordDims>::value);
                const auto dynamicSubIndex = loadBeginOffset(UnresolvedBeforeDynamic{}) + dynamicArrayExtents[0];
                return computeRecursive<nextSubMappingIndex>(
                    Cat<ResolvedSoFar, RecordCoord<dynamic>>{},
                    UnresolvedAfterDynamic{},
                    llama::ArrayDims{dynamicSubIndex},
                    pop_front(dynamicArrayExtents),
                    blobs);
            }
        }

        template <
            std::size_t MappingIndex,
            typename RecordCoordBeforeThisMapping,
            typename RecordCoordForThisMapping,
            typename Blob>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto mapToAddress(
            RecordCoordBeforeThisMapping,
            RecordCoordForThisMapping,
            ArrayDims coord,
            Array<Blob, blobCount>& blobs) const -> std::byte*
        {
            // we need to shift the record coord before mapping, because the user exposed RecordDim contains the
            // artificial EndOffset and Size fields, which the RecordDim of the submappings don't have.
            using ExposedSubRecordDim = GetType<RecordDim, RecordCoordBeforeThisMapping>;
            using ShiftedCoord =
                typename internal::ShiftRecordCoord<ExposedSubRecordDim, RecordCoordForThisMapping>::Coord;
            auto [nr, offset] = blobNrAndOffset(get<MappingIndex>(subMappings), ShiftedCoord{}, coord);
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<MappingIndex>>(
                [nr = std::ref(nr)](auto i) { nr += boost::mp11::mp_at<MappedSubRecordDims, decltype(i)>::blobCount; });
            return &blobs[nr][offset];
        }

        template <typename Mapping, std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            const Mapping& mapping,
            RecordCoord<RecordCoords...>,
            ArrayDims coord) const -> NrAndOffset
        {
            return mapping.template blobNrAndOffset<RecordCoords...>(coord);
        }
    };
} // namespace llama::mapping
