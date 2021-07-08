#pragma once

#include "Common.hpp"

#include <atomic>
#include <boost/core/demangle.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama::mapping
{
    namespace internal
    {
        template <typename RecordDim, std::size_t... Coords>
        auto coordName(RecordCoord<Coords...>) -> std::string
        {
            using Tags = GetTags<RecordDim, RecordCoord<Coords...>>;

            std::string r;
            boost::mp11::mp_for_each<Tags>(
                [&](auto tag)
                {
                    if (!r.empty())
                        r += '.';
                    r += structName(tag);
                });
            return r;
        }
    } // namespace internal

    /// Forwards all calls to the inner mapping. Traces all accesses made through this mapping and prints a summary on
    /// destruction.
    /// \tparam Mapping The type of the inner mapping.
    template <typename Mapping>
    struct Trace
    {
        using ArrayDims = typename Mapping::ArrayDims;
        using RecordDim = typename Mapping::RecordDim;
        static constexpr std::size_t blobCount = Mapping::blobCount;

        constexpr Trace() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Trace(Mapping mapping) : mapping(mapping)
        {
            forEachLeaf<RecordDim>([&](auto coord) { fieldHits[internal::coordName<RecordDim>(coord)] = 0; });
        }

        Trace(const Trace&) = delete;
        auto operator=(const Trace&) -> Trace& = delete;

        Trace(Trace&&) noexcept = default;
        auto operator=(Trace&&) noexcept -> Trace& = default;

        ~Trace()
        {
            if (!fieldHits.empty())
            {
                std::cout << "Trace mapping, number of accesses:\n";
                for (const auto& [k, v] : fieldHits)
                    std::cout << '\t' << k << ":\t" << v << '\n';
            }
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return mapping.arrayDims();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.blobSize(i);
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayDims coord, Array<std::size_t, N> dynamicArrayExtents = {})
            const -> NrAndOffset
        {
            const static auto name = internal::coordName<RecordDim>(RecordCoord<RecordCoords...>{});
            fieldHits.at(name)++;

            LLAMA_FORCE_INLINE_RECURSIVE return mapping.template blobNrAndOffset<RecordCoords...>(coord);
        }

        Mapping mapping;
        mutable std::unordered_map<std::string, std::atomic<std::size_t>> fieldHits;
    };
} // namespace llama::mapping
