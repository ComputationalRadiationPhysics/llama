// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"

namespace llama::mapping
{
    /// Maps all ArrayDims coordinates into the same location and layouts struct members consecutively. This mapping is
    /// used for temporary, single element views.
    template <typename T_ArrayDims, typename T_RecordDim>
    struct One
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;

        static constexpr std::size_t blobCount = 1;

        constexpr One() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr One(ArrayDims, RecordDim = {})
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            // TODO: not sure if this is the right approach, since we take any ArrayDims in the ctor
            ArrayDims ad;
            for (auto i = 0; i < ArrayDims::rank; i++)
                ad[i] = 1;
            return ad;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return sizeOf<RecordDim>;
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims, Array<std::size_t, N> = {}) const
            -> NrAndOffset
        {
            constexpr auto offset = offsetOf<RecordDim, RecordCoord<RecordCoords...>>;
            return {0, offset};
        }
    };
} // namespace llama::mapping
