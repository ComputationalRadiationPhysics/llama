// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// The maximum number of vector lanes that can be used to fetch each leaf type in the record dimension into a
    /// vector register of the given size in bits.
    template <typename RecordDim, std::size_t VectorRegisterBits>
    inline constexpr std::size_t maxLanes = []() constexpr
    {
        auto max = std::numeric_limits<std::size_t>::max();
        forEachLeaf<RecordDim>(
            [&](auto coord)
            {
                using AttributeType = GetType<RecordDim, decltype(coord)>;
                max = std::min(max, VectorRegisterBits / (sizeof(AttributeType) * CHAR_BIT));
            });
        return max;
    }
    ();

    /// Array of struct of arrays mapping. Used to create a \ref View via \ref allocView.
    /// \tparam Lanes The size of the inner arrays of this array of struct of arrays.
    /// \tparam T_LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        std::size_t Lanes,
        typename T_LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct AoSoA
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        using LinearizeArrayDimsFunctor = T_LinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount = 1;

        constexpr AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr AoSoA(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim>;
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents = {}) const -> NrAndOffset
        {
            const auto flatArrayIndex = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize);
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<RecordDim> * Lanes) * blockIndex
                + offsetOf<RecordDim, RecordCoord<RecordCoords...>> * Lanes
                + sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>) * laneIndex;
            return {0, offset};
        }

    private:
        ArrayDims arrayDimsSize;
    };

    template <std::size_t Lanes, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredAoSoA
    {
        template <typename ArrayDims, typename RecordDim>
        using type = AoSoA<ArrayDims, RecordDim, Lanes, LinearizeArrayDimsFunctor>;
    };
} // namespace llama::mapping
