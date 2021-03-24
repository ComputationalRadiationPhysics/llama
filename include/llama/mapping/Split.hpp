#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template <typename... DatumElements, std::size_t FirstCoord, std::size_t... Coords>
        auto partitionDatumDomain(DatumStruct<DatumElements...>, DatumCoord<FirstCoord, Coords...>)
        {
            using namespace boost::mp11;
            if constexpr (sizeof...(Coords) == 0)
            {
                using With = DatumStruct<mp_at_c<DatumStruct<DatumElements...>, FirstCoord>>;
                using Without = mp_erase_c<DatumStruct<DatumElements...>, FirstCoord, FirstCoord + 1>;
                return mp_list<With, Without>{};
            }
            else
            {
                using Result = decltype(partitionDatumDomain(
                    DatumStruct<mp_at_c<DatumStruct<DatumElements...>, FirstCoord>>{},
                    DatumCoord<Coords...>{}));
                using With = mp_replace_at_c<DatumStruct<DatumElements...>, FirstCoord, mp_first<Result>>;
                using Without = mp_replace_at_c<DatumStruct<DatumElements...>, FirstCoord, mp_second<Result>>;
                return mp_list<With, Without>{};
            }
        }

        template <
            std::size_t FirstDstCoord,
            std::size_t... DstCoords,
            std::size_t FirstSkippedCoord,
            std::size_t... SkippedCoords>
        constexpr auto offsetCoord(
            DatumCoord<FirstDstCoord, DstCoords...>,
            DatumCoord<FirstSkippedCoord, SkippedCoords...>)
        {
            if constexpr (FirstDstCoord < FirstSkippedCoord)
                return DatumCoord<FirstDstCoord, DstCoords...>{};
            else if constexpr (FirstDstCoord > FirstSkippedCoord)
                return DatumCoord<FirstDstCoord - 1, DstCoords...>{};
            else
                return cat(
                    DatumCoord<FirstDstCoord>{},
                    offsetCoord(DatumCoord<DstCoords...>{}, DatumCoord<SkippedCoords...>{}));
        }
    } // namespace internal

    /// Mapping which splits off a part of the datum domain and maps it differently then the rest.
    /// \tparam DatumCoordForMapping1 A \ref DatumCoord selecting the part of the datum domain to be mapped differently.
    /// \tparam MappingTemplate1 The mapping used for the selected part of the datum domain.
    /// \tparam MappingTemplate2 The mapping used for the not selected part of the datum domain.
    /// \tparam SeparateBlobs If true, both pieces of the datum domain are mapped to separate blobs.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename DatumCoordForMapping1,
        template <typename...>
        typename MappingTemplate1,
        template <typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct Split
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;

        using DatumDomainPartitions = decltype(internal::partitionDatumDomain(DatumDomain{}, DatumCoordForMapping1{}));
        using DatumDomain1 = boost::mp11::mp_first<DatumDomainPartitions>;
        using DatumDomain2 = boost::mp11::mp_second<DatumDomainPartitions>;

        using Mapping1 = MappingTemplate1<ArrayDomain, DatumDomain1>;
        using Mapping2 = MappingTemplate2<ArrayDomain, DatumDomain2>;

        static constexpr std::size_t blobCount = SeparateBlobs ? Mapping1::blobCount + Mapping2::blobCount : 1;
        static_assert(SeparateBlobs || Mapping1::blobCount == 1);
        static_assert(SeparateBlobs || Mapping2::blobCount == 1);

        constexpr Split() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Split(ArrayDomain size) : arrayDomainSize(size), mapping1(size), mapping2(size)
        {
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

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            // print_type_in_compilation_error<DatumDomain1>();
            if constexpr (DatumCoordCommonPrefixIsSame<DatumCoordForMapping1, DatumCoord<DatumDomainCoord...>>)
            {
                using namespace boost::mp11;
                // zero all coordinate values that are part of DatumCoordForMapping1
                constexpr auto prefixLength = DatumCoordForMapping1::size;
                using Prefix = mp_repeat_c<mp_list_c<std::size_t, 0>, DatumCoordForMapping1::size>;
                using Suffix = mp_drop_c<mp_list_c<std::size_t, DatumDomainCoord...>, DatumCoordForMapping1::size>;
                return blobNrAndOffset(DatumCoordFromList<mp_append<Prefix, Suffix>>{}, coord, mapping1);
            }
            else
            {
                constexpr auto dstCoord
                    = internal::offsetCoord(DatumCoord<DatumDomainCoord...>{}, DatumCoordForMapping1{});
                auto nrAndOffset = blobNrAndOffset(dstCoord, coord, mapping2);
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
        template <std::size_t... DatumDomainCoord, typename Mapping>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            DatumCoord<DatumDomainCoord...>,
            ArrayDomain coord,
            const Mapping& mapping) const -> NrAndOffset
        {
            return mapping.template blobNrAndOffset<DatumDomainCoord...>(coord);
        }

    public:
        ArrayDomain arrayDomainSize = {};
        Mapping1 mapping1;
        Mapping2 mapping2;
    };

    template <
        typename DatumCoordForMapping1,
        template <typename...>
        typename MappingTemplate1,
        template <typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct PreconfiguredSplit
    {
        template <typename ArrayDomain, typename DatumDomain>
        using type
            = Split<ArrayDomain, DatumDomain, DatumCoordForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>;
    };
} // namespace llama::mapping
