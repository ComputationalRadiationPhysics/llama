#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template <typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
        auto partitionRecordDim(Record<Fields...>, RecordCoord<FirstCoord, Coords...>)
        {
            using namespace boost::mp11;
            if constexpr (sizeof...(Coords) == 0)
            {
                using With = Record<mp_at_c<Record<Fields...>, FirstCoord>>;
                using Without = mp_erase_c<Record<Fields...>, FirstCoord, FirstCoord + 1>;
                return mp_list<With, Without>{};
            }
            else
            {
                using Result = decltype(partitionRecordDim(
                    Record<mp_at_c<Record<Fields...>, FirstCoord>>{},
                    RecordCoord<Coords...>{}));
                using With = mp_replace_at_c<Record<Fields...>, FirstCoord, mp_first<Result>>;
                using Without = mp_replace_at_c<Record<Fields...>, FirstCoord, mp_second<Result>>;
                return mp_list<With, Without>{};
            }
        }

        template <
            std::size_t FirstDstCoord,
            std::size_t... DstCoords,
            std::size_t FirstSkippedCoord,
            std::size_t... SkippedCoords>
        constexpr auto offsetCoord(
            RecordCoord<FirstDstCoord, DstCoords...>,
            RecordCoord<FirstSkippedCoord, SkippedCoords...>)
        {
            if constexpr (FirstDstCoord < FirstSkippedCoord)
                return RecordCoord<FirstDstCoord, DstCoords...>{};
            else if constexpr (FirstDstCoord > FirstSkippedCoord)
                return RecordCoord<FirstDstCoord - 1, DstCoords...>{};
            else
                return cat(
                    RecordCoord<FirstDstCoord>{},
                    offsetCoord(RecordCoord<DstCoords...>{}, RecordCoord<SkippedCoords...>{}));
        }
    } // namespace internal

    /// Mapping which splits off a part of the record dimension and maps it differently then the rest.
    /// \tparam RecordCoordForMapping1 A \ref RecordCoord selecting the part of the record dimension to be mapped
    /// differently.
    /// \tparam MappingTemplate1 The mapping used for the selected part of the record dimension.
    /// \tparam MappingTemplate2 The mapping used for the not selected part of the record dimension.
    /// \tparam SeparateBlobs If true, both pieces of the record dimension are mapped to separate blobs.
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        typename RecordCoordForMapping1,
        template <typename...>
        typename MappingTemplate1,
        template <typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct Split
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;

        using RecordDimPartitions = decltype(internal::partitionRecordDim(RecordDim{}, RecordCoordForMapping1{}));
        using RecordDim1 = boost::mp11::mp_first<RecordDimPartitions>;
        using RecordDim2 = boost::mp11::mp_second<RecordDimPartitions>;

        using Mapping1 = MappingTemplate1<ArrayDims, RecordDim1>;
        using Mapping2 = MappingTemplate2<ArrayDims, RecordDim2>;

        static constexpr std::size_t blobCount = SeparateBlobs ? Mapping1::blobCount + Mapping2::blobCount : 1;
        static_assert(SeparateBlobs || Mapping1::blobCount == 1);
        static_assert(SeparateBlobs || Mapping2::blobCount == 1);

        constexpr Split() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Split(ArrayDims size) : mapping1(size), mapping2(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return mapping1.arrayDims();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
        {
            if constexpr (SeparateBlobs)
            {
                if (i < Mapping1::blobCount)
                    return mapping1.blobSize(i);
                else
                    return mapping2.blobSize(i - Mapping1::blobCount);
            }
            else
                return mapping1.blobSize(0) + mapping2.blobSize(0);
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents = {}) const -> NrAndOffset
        {
            if constexpr (RecordCoordCommonPrefixIsSame<RecordCoordForMapping1, RecordCoord<RecordCoords...>>)
            {
                using namespace boost::mp11;
                // zero all coordinate values that are part of RecordCoordForMapping1
                using Prefix = mp_repeat_c<mp_list_c<std::size_t, 0>, RecordCoordForMapping1::size>;
                using Suffix = mp_drop_c<mp_list_c<std::size_t, RecordCoords...>, RecordCoordForMapping1::size>;
                return blobNrAndOffset(
                    RecordCoordFromList<mp_append<Prefix, Suffix>>{},
                    coord,
                    dynamicArrayExtents,
                    mapping1);
            }
            else
            {
                constexpr auto dstCoord
                    = internal::offsetCoord(RecordCoord<RecordCoords...>{}, RecordCoordForMapping1{});
                auto nrAndOffset = blobNrAndOffset(dstCoord, coord, dynamicArrayExtents, mapping2);
                if constexpr (SeparateBlobs)
                    nrAndOffset.nr += Mapping1::blobCount;
                else
                {
                    for (auto i = 0; i < Mapping1::blobCount; i++)
                        nrAndOffset.offset += mapping1.blobSize(i);
                }
                return nrAndOffset;
            }
        }

    private:
        template <std::size_t... RecordCoords, std::size_t N, typename Mapping>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            RecordCoord<RecordCoords...>,
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents,
            const Mapping& mapping) const -> NrAndOffset
        {
            return mapping.template blobNrAndOffset<RecordCoords...>(coord, dynamicArrayExtents);
        }

    public:
        Mapping1 mapping1;
        Mapping2 mapping2;
    };

    template <
        typename RecordCoordForMapping1,
        template <typename...>
        typename MappingTemplate1,
        template <typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct PreconfiguredSplit
    {
        template <typename ArrayDims, typename RecordDim>
        using type
            = Split<ArrayDims, RecordDim, RecordCoordForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>;
    };
} // namespace llama::mapping
