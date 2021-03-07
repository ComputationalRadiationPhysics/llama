// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tightly packed.
    /// \tparam T_LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattended. See \ref
    /// FlattenRecordDimInOrder and \ref FlattenRecordDimMinimizePadding.
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        bool AlignAndPad = true,
        typename T_LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template <typename> typename FlattenRecordDim = FlattenRecordDimInOrder>
    struct AoS
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        using LinearizeArrayDimsFunctor = T_LinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount = 1;

        constexpr AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr AoS(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDimsFunctor{}.size(arrayDimsSize)
                * flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad>;
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            ArrayDims coord,
            Array<std::size_t, N> dynamicArrayExtents = {}) const -> NrAndOffset
        {
            constexpr std::size_t flatIndex =
#ifdef __NVCC__
                Flattener{}.template flatIndex<RecordCoords...>;
#else
                Flattener::template flatIndex<RecordCoords...>;
#endif
            const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                    * flatSizeOf<typename Flattener::FlatRecordDim,
                                 AlignAndPad> + flatOffsetOf<typename Flattener::FlatRecordDim, flatIndex, AlignAndPad>;
            return {0, offset};
        }

    private:
        using Flattener = FlattenRecordDim<T_RecordDim>;
        ArrayDims arrayDimsSize;
    };

    /// Array of struct mapping preserving the alignment of the field types by inserting padding.
    /// \see AoS
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using AlignedAoS = AoS<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

    /// Array of struct mapping preserving the alignment of the field types by inserting padding and permuting the field
    /// order to minimize this padding. \see AoS
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MinAlignedAoS = AoS<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor, FlattenRecordDimMinimizePadding>;

    /// Array of struct mapping packing the field types tightly, violating the types alignment requirements.
    /// \see AoS
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using PackedAoS = AoS<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

    template <bool AlignAndPad = true, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredAoS
    {
        template <typename ArrayDims, typename RecordDim>
        using type = AoS<ArrayDims, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor>;
    };
} // namespace llama::mapping
