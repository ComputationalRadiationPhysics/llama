#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

TEST_CASE("Split.SoA.AoS.1Buffer")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto array_dims = ArrayDims{16, 16};

    // we layout Pos as SoA, the rest as AoS
    auto mapping = llama::mapping::
        Split<ArrayDims, Particle, llama::RecordCoord<0>, llama::mapping::SingleBlobSoA, llama::mapping::PackedAoS>{
            array_dims};

    constexpr auto mapping1_size = 6120;
    const auto coord = ArrayDims{0, 0};
    CHECK(mapping.blobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{0, 2048});
    CHECK(mapping.blobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{0, 4096});
    CHECK(mapping.blobNrAndOffset<1>(coord) == llama::NrAndOffset{0, mapping1_size + 24});
    CHECK(mapping.blobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{0, mapping1_size + 28});
    CHECK(mapping.blobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{0, mapping1_size + 36});
    CHECK(mapping.blobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{0, mapping1_size + 44});
    CHECK(mapping.blobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{0, mapping1_size + 52});
    CHECK(mapping.blobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{0, mapping1_size + 53});
    CHECK(mapping.blobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{0, mapping1_size + 54});
    CHECK(mapping.blobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{0, mapping1_size + 55});
}

TEST_CASE("Split.AoSoA8.AoS.One.SoA.4Buffer")
{
    // split out momentum as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    using ArrayDims = llama::ArrayDims<1>;
    auto array_dims = ArrayDims{32};
    auto mapping = llama::mapping::Split<
        ArrayDims,
        Particle,
        llama::RecordCoord<2>,
        llama::mapping::PreconfiguredAoSoA<8>::type,
        llama::mapping::PreconfiguredSplit<
            llama::RecordCoord<1>,
            llama::mapping::One,
            llama::mapping::PreconfiguredSplit<
                llama::RecordCoord<0>,
                llama::mapping::PackedAoS,
                llama::mapping::SingleBlobSoA,
                true>::type,
            true>::type,
        true>{array_dims};

    CHECK(mapping.blobNrAndOffset<0, 0>({0}) == llama::NrAndOffset{2, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>({0}) == llama::NrAndOffset{2, 8});
    CHECK(mapping.blobNrAndOffset<0, 2>({0}) == llama::NrAndOffset{2, 16});
    CHECK(mapping.blobNrAndOffset<1>({0}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({0}) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<2, 1>({0}) == llama::NrAndOffset{0, 64});
    CHECK(mapping.blobNrAndOffset<2, 2>({0}) == llama::NrAndOffset{0, 128});
    CHECK(mapping.blobNrAndOffset<3, 0>({0}) == llama::NrAndOffset{3, 0});
    CHECK(mapping.blobNrAndOffset<3, 1>({0}) == llama::NrAndOffset{3, 32});
    CHECK(mapping.blobNrAndOffset<3, 2>({0}) == llama::NrAndOffset{3, 64});
    CHECK(mapping.blobNrAndOffset<3, 3>({0}) == llama::NrAndOffset{3, 96});

    CHECK(mapping.blobNrAndOffset<0, 0>({31}) == llama::NrAndOffset{2, 744});
    CHECK(mapping.blobNrAndOffset<0, 1>({31}) == llama::NrAndOffset{2, 752});
    CHECK(mapping.blobNrAndOffset<0, 2>({31}) == llama::NrAndOffset{2, 760});
    CHECK(mapping.blobNrAndOffset<1>({31}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({31}) == llama::NrAndOffset{0, 632});
    CHECK(mapping.blobNrAndOffset<2, 1>({31}) == llama::NrAndOffset{0, 696});
    CHECK(mapping.blobNrAndOffset<2, 2>({31}) == llama::NrAndOffset{0, 760});
    CHECK(mapping.blobNrAndOffset<3, 0>({31}) == llama::NrAndOffset{3, 31});
    CHECK(mapping.blobNrAndOffset<3, 1>({31}) == llama::NrAndOffset{3, 63});
    CHECK(mapping.blobNrAndOffset<3, 2>({31}) == llama::NrAndOffset{3, 95});
    CHECK(mapping.blobNrAndOffset<3, 3>({31}) == llama::NrAndOffset{3, 127});

    // std::ofstream{"Split.AoSoA8.AoS.One.SoA.4Buffer.svg"} << llama::toSvg(mapping);
}
