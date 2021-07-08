// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam SeparateBuffers If true, every element of the record dimension is mapped to its own buffer.
    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        bool SeparateBuffers = true,
        typename T_LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct SoA
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        using LinearizeArrayDimsFunctor = T_LinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount
            = SeparateBuffers ? boost::mp11::mp_size<FlatRecordDim<RecordDim>>::value : 1;

        constexpr SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr SoA(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t blobIndex) const -> std::size_t
        {
            if constexpr (SeparateBuffers)
            {
                constexpr Array<std::size_t, blobCount> typeSizes = []() constexpr
                {
                    Array<std::size_t, blobCount> r{};
                    forEachLeaf<RecordDim>([&r, i = 0](auto coord) mutable constexpr
                                           { r[i++] = sizeof(GetType<RecordDim, decltype(coord)>); });
                    return r;
                }
                ();
                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * typeSizes[blobIndex];
            }
            else
                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim>;
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents = {}) const -> NrAndOffset
        {
            if constexpr (SeparateBuffers)
            {
                constexpr auto blob = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                    * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>);
                return {blob, offset};
            }
            else
            {
                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                        * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>)
                    + offsetOf<
                          RecordDim,
                          RecordCoord<RecordCoords...>> * LinearizeArrayDimsFunctor{}.size(arrayDimsSize);
                return {0, offset};
            }
        }

    private:
        ArrayDims arrayDimsSize;
    };

    /// Struct of array mapping storing the entire layout in a single blob.
    /// \see SoA
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using SingleBlobSoA = SoA<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
    /// \see SoA
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MultiBlobSoA = SoA<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

    template <bool SeparateBuffers = true, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredSoA
    {
        template <typename ArrayDims, typename RecordDim>
        using type = SoA<ArrayDims, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>;
    };
} // namespace llama::mapping
