// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "ArrayDimsIndexRange.hpp"
#include "Core.hpp"

#include <boost/functional/hash.hpp>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace llama
{
    namespace internal
    {
        template <std::size_t... Coords>
        auto toVec(RecordCoord<Coords...>) -> std::vector<std::size_t>
        {
            return {Coords...};
        }

        template <typename Tag>
        auto tagToString(Tag tag)
        {
            return structName(tag);
        }

        // handle array indices
        template <std::size_t N>
        auto tagToString(RecordCoord<N>)
        {
            return std::to_string(N);
        }

        template <typename RecordDim, std::size_t... CoordsBefore, std::size_t CoordCurrent, std::size_t... CoordsAfter>
        void collectTagsAsStrings(
            std::vector<std::string>& v,
            RecordCoord<CoordsBefore...> before,
            RecordCoord<CoordCurrent, CoordsAfter...> after)
        {
            using Tag = GetTag<RecordDim, RecordCoord<CoordsBefore..., CoordCurrent>>;
            v.push_back(tagToString(Tag{}));
            if constexpr (sizeof...(CoordsAfter) > 0)
                collectTagsAsStrings<RecordDim>(
                    v,
                    RecordCoord<CoordsBefore..., CoordCurrent>{},
                    RecordCoord<CoordsAfter...>{});
        }

        template <typename RecordDim, std::size_t... Coords>
        auto tagsAsStrings(RecordCoord<Coords...>) -> std::vector<std::string>
        {
            std::vector<std::string> v;
            collectTagsAsStrings<RecordDim>(v, RecordCoord<>{}, RecordCoord<Coords...>{});
            return v;
        }

        template <typename Mapping, typename ArrayDims, std::size_t... Coords>
        auto mappingBlobNrAndOffset(const Mapping& mapping, const ArrayDims& adCoord, RecordCoord<Coords...>)
        {
            return mapping.template blob_nr_and_offset<Coords...>(adCoord);
        }

        inline auto color(const std::vector<std::size_t>& recordCoord) -> std::size_t
        {
            auto c = boost::hash<std::vector<std::size_t>>{}(recordCoord) &0xFFFFFF;
            const auto channelSum = ((c & 0xFF0000) >> 4) + ((c & 0xFF00) >> 2) + c & 0xFF;
            if (channelSum < 200)
                c |= 0x404040; // ensure color per channel is at least 0x40.
            return c;
        }

        template <std::size_t Dim>
        auto formatUdCoord(const ArrayDims<Dim>& coord)
        {
            if constexpr (Dim == 1)
                return std::to_string(coord[0]);
            else
            {
                std::string s = "{";
                for (auto v : coord)
                {
                    if (s.size() >= 2)
                        s += ",";
                    s += std::to_string(v);
                }
                s += "}";
                return s;
            }
        }

        inline auto formatDDTags(const std::vector<std::string>& tags)
        {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += ".";
                s += tag;
            }
            return s;
        }

        template <std::size_t Dim>
        struct FieldBox
        {
            ArrayDims<Dim> adCoord;
            std::vector<std::size_t> recordCoord;
            std::vector<std::string> recordTags;
            NrAndOffset nrAndOffset;
            std::size_t size;
        };

        template <typename Mapping>
        auto boxesFromMapping(const Mapping& mapping) -> std::vector<FieldBox<Mapping::ArrayDims::rank>>
        {
            using ArrayDims = typename Mapping::ArrayDims;
            using RecordDim = typename Mapping::RecordDim;

            std::vector<FieldBox<Mapping::ArrayDims::rank>> infos;

            for (auto adCoord : ArrayDimsIndexRange{mapping.array_dims()})
            {
                forEachLeaf<RecordDim>(
                    [&](auto coord)
                    {
                        constexpr int size = sizeof(GetType<RecordDim, decltype(coord)>);
                        infos.push_back(
                            {adCoord,
                             internal::toVec(coord),
                             internal::tagsAsStrings<RecordDim>(coord),
                             internal::mappingBlobNrAndOffset(mapping, adCoord, coord),
                             size});
                    });
            }

            return infos;
        }

        template <std::size_t Dim>
        auto breakBoxes(std::vector<FieldBox<Dim>> boxes, std::size_t wrapByteCount) -> std::vector<FieldBox<Dim>>
        {
            for (std::size_t i = 0; i < boxes.size(); i++)
            {
                auto& fb = boxes[i];
                if (fb.nrAndOffset.offset / wrapByteCount != (fb.nrAndOffset.offset + fb.size - 1) / wrapByteCount)
                {
                    const auto remainingSpace = wrapByteCount - fb.nrAndOffset.offset % wrapByteCount;
                    auto newFb = fb;
                    newFb.nrAndOffset.offset = fb.nrAndOffset.offset + remainingSpace;
                    newFb.size = fb.size - remainingSpace;
                    fb.size = remainingSpace;
                    boxes.push_back(newFb);
                }
            }
            return boxes;
        }
    } // namespace internal

    /// Returns an SVG image visualizing the memory layout created by the given
    /// mapping. The created memory blocks are wrapped after wrapByteCount
    /// bytes.
    template <typename Mapping>
    auto toSvg(const Mapping& mapping, std::size_t wrapByteCount = 64, bool breakBoxes = true) -> std::string
    {
        constexpr auto byteSizeInPixel = 30;
        constexpr auto blobBlockWidth = 60;

        auto infos = internal::boxesFromMapping(mapping);
        if (breakBoxes)
            infos = internal::breakBoxes(std::move(infos), wrapByteCount);

        std::string svg;

        std::array<int, Mapping::blob_count + 1> blobYOffset{};
        for (auto i = 0; i < Mapping::blob_count; i++)
        {
            const auto blobRows = (mapping.blob_size(i) + wrapByteCount - 1) / wrapByteCount;
            blobYOffset[i + 1] = blobYOffset[i] + (blobRows + 1) * byteSizeInPixel; // one row gap between blobs
            const auto height = blobRows * byteSizeInPixel;
            svg += fmt::format(
                R"a(<rect x="0" y="{}" width="{}" height="{}" fill="#AAA" stroke="#000"/>
<text x="{}" y="{}" fill="#000" text-anchor="middle">Blob: {}</text>
)a",
                blobYOffset[i],
                blobBlockWidth,
                height,
                blobBlockWidth / 2,
                blobYOffset[i] + height / 2,
                i);
        }

        svg = fmt::format(
                  R"(<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .label {{ font: {}px sans-serif; }}
    </style>
)",
                  blobBlockWidth + wrapByteCount * byteSizeInPixel,
                  blobYOffset.back() - byteSizeInPixel,
                  byteSizeInPixel / 2)
            + svg;

        for (const auto& info : infos)
        {
            const auto blobY = blobYOffset[info.nrAndOffset.nr];
            auto x = (info.nrAndOffset.offset % wrapByteCount) * byteSizeInPixel + blobBlockWidth;
            auto y = (info.nrAndOffset.offset / wrapByteCount) * byteSizeInPixel + blobY;
            const auto fill = internal::color(info.recordCoord);
            const auto width = byteSizeInPixel * info.size;

            constexpr auto cropBoxes = true;
            if (cropBoxes)
            {
                svg += fmt::format(
                    R"(<svg x="{}" y="{}" width="{}" height="{}">
)",
                    x,
                    y,
                    width,
                    byteSizeInPixel);
                x = 0;
                y = 0;
            }
            svg += fmt::format(
                R"(<rect x="{}" y="{}" width="{}" height="{}" fill="#{:X}" stroke="#000"/>
)",
                x,
                y,
                width,
                byteSizeInPixel,
                fill);
            for (auto i = 1; i < info.size; i++)
            {
                svg += fmt::format(
                    R"(<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#777"/>
)",
                    x + i * byteSizeInPixel,
                    y + byteSizeInPixel * 2 / 3,
                    x + i * byteSizeInPixel,
                    y + byteSizeInPixel);
            }
            svg += fmt::format(
                R"(<text x="{}" y="{}" fill="#000" text-anchor="middle" class="label">{} {}</text>
)",
                x + width / 2,
                y + byteSizeInPixel * 3 / 4,
                internal::formatUdCoord(info.adCoord),
                internal::formatDDTags(info.recordTags));
            if (cropBoxes)
                svg += R"(</svg>
)";
        }
        svg += "</svg>";
        return svg;
    }

    /// Returns an HTML document visualizing the memory layout created by the
    /// given mapping. The visualization is resizeable.
    template <typename Mapping>
    auto toHtml(const Mapping& mapping) -> std::string
    {
        constexpr auto byteSizeInPixel = 30;
        constexpr auto rulerLengthInBytes = 512;
        constexpr auto rulerByteInterval = 8;

        auto infos = internal::boxesFromMapping(mapping);
        std::stable_sort(
            begin(infos),
            end(infos),
            [](const auto& a, const auto& b) {
                return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset)
                    < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
            });
        infos.erase(
            std::unique(
                begin(infos),
                end(infos),
                [](const auto& a, const auto& b) { return a.nrAndOffset == b.nrAndOffset; }),
            end(infos));

        auto cssClass = [](const std::vector<std::string>& tags)
        {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += "_";
                s += tag;
            }
            return s;
        };

        std::string svg;
        svg += fmt::format(
            R"(<!DOCTYPE html>
<html>
<head>
<style>
.box {{
    outline: 1px solid;
    display: inline-block;
    white-space: nowrap;
    height: {}px;
    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #777 29px, #777 30px);
    text-align: center;
    overflow: hidden;
    vertical-align: middle;
}}
#ruler {{
    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #000 29px, #000 30px);
    border-bottom: 1px solid;
    height: 20px;
    margin-bottom: 20px;
}}
#ruler div {{
    position: absolute;
    display: inline-block;
}}
)",
            byteSizeInPixel);
        using RecordDim = typename Mapping::RecordDim;
        forEachLeaf<RecordDim>(
            [&](auto coord)
            {
                constexpr int size = sizeof(GetType<RecordDim, decltype(coord)>);

                svg += fmt::format(
                    R"(.{} {{
    width: {}px;
    background-color: #{:X};
}}
)",
                    cssClass(internal::tagsAsStrings<RecordDim>(coord)),
                    byteSizeInPixel * size,
                    internal::color(internal::toVec(coord)));
            });

        svg += fmt::format(R"(</style>
</head>
<body>
    <header id="ruler">
)");
        for (auto i = 0; i < rulerLengthInBytes; i += rulerByteInterval)
            svg += fmt::format(
                R"(</style>
        <div style="margin-left: {}px;">{}</div>)",
                i * byteSizeInPixel,
                i);
        svg += fmt::format(R"(
    </header>
)");

        auto currentBlobNr = std::numeric_limits<std::size_t>::max();
        for (const auto& info : infos)
        {
            if (currentBlobNr != info.nrAndOffset.nr)
            {
                currentBlobNr = info.nrAndOffset.nr;
                svg += fmt::format("<h1>Blob: {}</h1>", currentBlobNr);
            }
            const auto width = byteSizeInPixel * info.size;
            svg += fmt::format(
                R"(<div class="box {0}" title="{1} {2}">{1} {2}</div>)",
                cssClass(info.recordTags),
                internal::formatUdCoord(info.adCoord),
                internal::formatDDTags(info.recordTags));
        }
        svg += R"(</body>
</html>)";
        return svg;
    }
} // namespace llama
