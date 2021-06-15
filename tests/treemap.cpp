#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

namespace tree = llama::mapping::tree;

namespace tag
{
    auto toString(Pos)
    {
        return "Pos";
    }
    auto toString(X)
    {
        return "X";
    }
    auto toString(Y)
    {
        return "Y";
    }
    auto toString(Z)
    {
        return "Z";
    }
    auto toString(Vel)
    {
        return "Vel";
    }
    auto toString(Mass)
    {
        return "Mass";
    }
    auto toString(Flags)
    {
        return "Flags";
    }
} // namespace tag

TEST_CASE("treemapping.empty")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 951);
    }
}

TEST_CASE("treemapping.Idem")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::Idem()};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 951);
    }
}

TEST_CASE("treemapping.LeafOnlyRT")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::LeafOnlyRT()};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "1C * [ 1C * [ 1C * Pos[ 256R * X(double) , 256R * Y(double) , 256R * Z(double) ] , 256R * Mass(float) , "
           "1C * Vel[ 256R * X(double) , 256R * Y(double) "
           ", 256R * Z(double) ] , 1C * Flags[ 256R * (bool) , 256R * (bool) , 256R * (bool) , 256R * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14096);
    }
}

TEST_CASE("treemapping.MoveRTDown<>")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::MoveRTDown<tree::TreeCoord<>>{4}};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "4R * [ 64R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 940);
    }
}

TEST_CASE("treemapping.MoveRTDown<0>")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::MoveRTDown<tree::TreeCoord<0>>{4}};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 4R * [ 4R * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 4R * Mass(float) , 4R * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 4R * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 112);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 120);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 128);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 136);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 144);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 152);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 992);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 1008);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 1016);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 1024);
    }
}

TEST_CASE("treemapping.MoveRTDown<0,0>")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::MoveRTDown<tree::TreeCoord<0, 0>>{4}};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 16R * [ 1R * Pos[ 4R * X(double) , 4R * Y(double) , 4R * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 32768);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 116);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 160);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 192);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 224);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 228);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 236);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 244);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 2048);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2080);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 2112);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 2144);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 2148);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 2156);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 2164);
    }
}

TEST_CASE("treemapping.MoveRTDownFixed<>")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::MoveRTDownFixed<tree::TreeCoord<>, 4>{}};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "4R * [ 64R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 940);
    }
}

TEST_CASE("treemapping.MoveRTDownFixed<0>")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::MoveRTDownFixed<tree::TreeCoord<0>, 4>{}};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 4R * [ 4R * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 4R * Mass(float) , 4R * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 4R * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 112);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 120);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 128);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 136);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 144);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 152);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 992);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 1008);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 1016);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 1024);
    }
}

TEST_CASE("treemapping.MoveRTDownFixed<0,0>")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{tree::functor::MoveRTDownFixed<tree::TreeCoord<0, 0>, 4>{}};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 16R * [ 1R * Pos[ 4R * X(double) , 4R * Y(double) , 4R * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 32768);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 116);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 160);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 192);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 224);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 228);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 236);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 244);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 2048);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2080);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 2112);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 2144);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 2148);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 2156);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 2164);
    }
}

TEST_CASE("treemapping.vectorblocks.runtime")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    const auto vector_width = 8;

    auto tree_operation_list = llama::Tuple{
        tree::functor::MoveRTDown<tree::TreeCoord<0>>{vector_width}, // move 8 down from ArrayDims (to
                                                                     // Position/Mass/Vel)
        tree::functor::MoveRTDown<tree::TreeCoord<0, 0>>{vector_width}, // move 8 down from Position (to X/Y/Z)
        tree::functor::MoveRTDown<tree::TreeCoord<0, 2>>{vector_width}, // move 8 down from Vel (to X/Y/Z)
    };
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 2R * [ 1R * Pos[ 8R * X(double) , 8R * Y(double) , 8R * Z(double) ] , 8R * Mass(float) , "
           "1R * Vel[ 8R * X(double) , 8R * Y(double) , 8R * Z(double) ] , 8R * Flags[ 1C * (bool) , 1C * (bool) "
           ", 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 192);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 224);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 288);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 352);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 136);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 196);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 232);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 296);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 360);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 960);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 1024);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 1088);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 1120);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 1184);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 1248);
    }
}

TEST_CASE("treemapping.vectorblocks.compiletime")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    constexpr auto vector_width = 8;

    auto tree_operation_list = llama::Tuple{
        tree::functor::MoveRTDownFixed<tree::TreeCoord<0>, vector_width>{}, // move 8 down from ArrayDims (to
                                                                            // Position/Mass/Vel)
        tree::functor::MoveRTDownFixed<tree::TreeCoord<0, 0>, vector_width>{}, // move 8 down from Position (to X/Y/Z)
        tree::functor::MoveRTDownFixed<tree::TreeCoord<0, 2>, vector_width>{}, // move 8 down from Vel (to X/Y/Z)
    };
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    CHECK(
        tree::toString(mapping.basicTree)
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "16R * [ 2R * [ 1R * Pos[ 8R * X(double) , 8R * Y(double) , 8R * Z(double) ] , 8R * Mass(float) , "
           "1R * Vel[ 8R * X(double) , 8R * Y(double) , 8R * Z(double) ] , 8R * Flags[ 1C * (bool) , 1C * (bool) "
           ", 1C * (bool) , 1C * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 14336);

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 192);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 224);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 288);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 352);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 136);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 196);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 232);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 296);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 360);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 960);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 1024);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 1088);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 1120);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 1184);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 1248);
    }
}

TEST_CASE("treemapping.getNode")
{
    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{16, 16};

    auto tree_operation_list = llama::Tuple{};
    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    using namespace tree;
    using namespace tree::functor::internal;

    CHECK(
        toString(getNode<TreeCoord<>>(mapping.resultTree))
        == "16R * [ 16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) "
           ", 1C * Vel[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * "
           "(bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        toString(getNode<TreeCoord<0>>(mapping.resultTree))
        == "16R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C "
           "* (bool) , 1C * (bool) ] ]");
    CHECK(
        toString(getNode<TreeCoord<0, 0>>(mapping.resultTree))
        == "1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ]");
    CHECK(toString(getNode<TreeCoord<0, 0, 0>>(mapping.resultTree)) == "1C * X(double)");
    CHECK(toString(getNode<TreeCoord<0, 0, 1>>(mapping.resultTree)) == "1C * Y(double)");
    CHECK(toString(getNode<TreeCoord<0, 0, 2>>(mapping.resultTree)) == "1C * Z(double)");
    CHECK(toString(getNode<TreeCoord<0, 1>>(mapping.resultTree)) == "1C * Mass(float)");
    CHECK(
        toString(getNode<TreeCoord<0, 2>>(mapping.resultTree))
        == "1C * Vel[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ]");
    CHECK(toString(getNode<TreeCoord<0, 2, 0>>(mapping.resultTree)) == "1C * X(double)");
    CHECK(toString(getNode<TreeCoord<0, 2, 1>>(mapping.resultTree)) == "1C * Y(double)");
    CHECK(toString(getNode<TreeCoord<0, 2, 2>>(mapping.resultTree)) == "1C * Z(double)");
    CHECK(
        toString(getNode<TreeCoord<0, 3>>(mapping.resultTree))
        == "1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ]");
    CHECK(toString(getNode<TreeCoord<0, 3, 0>>(mapping.resultTree)) == "1C * (bool)");
    CHECK(toString(getNode<TreeCoord<0, 3, 1>>(mapping.resultTree)) == "1C * (bool)");
    CHECK(toString(getNode<TreeCoord<0, 3, 2>>(mapping.resultTree)) == "1C * (bool)");
    CHECK(toString(getNode<TreeCoord<0, 3, 3>>(mapping.resultTree)) == "1C * (bool)");
}

TEST_CASE("treemapping")
{
    constexpr std::size_t array_dims_size = 12;

    using ArrayDims = llama::ArrayDims<2>;
    const ArrayDims array_dims{array_dims_size, array_dims_size};

    auto tree_operation_list = llama::Tuple{tree::functor::Idem(), tree::functor::LeafOnlyRT{}, tree::functor::Idem{}};

    using Mapping = tree::Mapping<ArrayDims, Particle, decltype(tree_operation_list)>;
    const Mapping mapping(array_dims, tree_operation_list);

    auto raw = prettyPrintType(mapping.basicTree);
#ifdef _WIN32
    replace_all(raw, "__int64", "long");
#endif
    const auto* const ref = R"(llama::mapping::tree::Node<
    llama::NoName,
    llama::Tuple<
        llama::mapping::tree::Node<
            llama::NoName,
            llama::Tuple<
                llama::mapping::tree::Node<
                    tag::Pos,
                    llama::Tuple<
                        llama::mapping::tree::Leaf<
                            tag::X,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Y,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Z,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::Leaf<
                    tag::Mass,
                    float,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::Node<
                    tag::Vel,
                    llama::Tuple<
                        llama::mapping::tree::Leaf<
                            tag::X,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Y,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Z,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::Node<
                    tag::Flags,
                    llama::Tuple<
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                0
                            >,
                            bool,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                1
                            >,
                            bool,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                2
                            >,
                            bool,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                3
                            >,
                            bool,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >
            >,
            unsigned long
        >
    >,
    unsigned long
>)";
    CHECK(raw == ref);

    auto raw2 = prettyPrintType(mapping.resultTree);
#ifdef _WIN32
    replace_all(raw2, "__int64", "long");
#endif
    const auto* const ref2 = R"(llama::mapping::tree::Node<
    llama::NoName,
    llama::Tuple<
        llama::mapping::tree::Node<
            llama::NoName,
            llama::Tuple<
                llama::mapping::tree::Node<
                    tag::Pos,
                    llama::Tuple<
                        llama::mapping::tree::Leaf<
                            tag::X,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Y,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Z,
                            double,
                            unsigned long
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::Leaf<
                    tag::Mass,
                    float,
                    unsigned long
                >,
                llama::mapping::tree::Node<
                    tag::Vel,
                    llama::Tuple<
                        llama::mapping::tree::Leaf<
                            tag::X,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Y,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            tag::Z,
                            double,
                            unsigned long
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::Node<
                    tag::Flags,
                    llama::Tuple<
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                0
                            >,
                            bool,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                1
                            >,
                            bool,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                2
                            >,
                            bool,
                            unsigned long
                        >,
                        llama::mapping::tree::Leaf<
                            llama::RecordCoord<
                                3
                            >,
                            bool,
                            unsigned long
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >
            >,
            std::integral_constant<
                unsigned long,
                1
            >
        >
    >,
    std::integral_constant<
        unsigned long,
        1
    >
>)";
    CHECK(raw2 == ref2);

    CHECK(
        tree::toString(mapping.basicTree)
        == "12R * [ 12R * [ 1C * Pos[ 1C * X(double) , 1C * Y(double) , 1C * Z(double) ] , 1C * Mass(float) , 1C * "
           "Vel[ 1C * X(double) , 1C * Y(double) , 1C * "
           "Z(double) ] , 1C * Flags[ 1C * (bool) , 1C * (bool) , 1C * (bool) , 1C * (bool) ] ] ]");
    CHECK(
        tree::toString(mapping.resultTree)
        == "1C * [ 1C * [ 1C * Pos[ 144R * X(double) , 144R * Y(double) , 144R * Z(double) ] , 144R * Mass(float) , "
           "1C * Vel[ 144R * X(double) , 144R * Y(double) , 144R * Z(double) ] , 1C * Flags[ 144R * (bool) , 144R "
           "* (bool) , 144R * (bool) , 144R * (bool) ] ] ]");

    CHECK(mapping.blobSize(0) == 8064);
    CHECK(mapping.blobNrAndOffset<2, 1>({50, 100}).offset == 10784);
    CHECK(mapping.blobNrAndOffset<2, 1>({50, 101}).offset == 10792);

    auto view = allocView(mapping);
    zeroStorage(view);

    for (size_t x = 0; x < array_dims[0]; ++x)
        for (size_t y = 0; y < array_dims[1]; ++y)
        {
            auto record = view(x, y);
            llama::forEachLeaf<Particle>([&](auto coord) { record(coord) = 0; }, tag::Vel{});
        }
    double sum = 0.0;
    for (size_t x = 0; x < array_dims[0]; ++x)
        for (size_t y = 0; y < array_dims[1]; ++y)
            sum += view({x, y})(llama::RecordCoord<0, 1>{});
    CHECK(sum == 0);
}

TEST_CASE("treeCoordToString")
{
    const auto coord = llama::ArrayDims<3>{6, 7, 8};
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<0, 0>>(coord))
        == "[ 6:0, 7:0, 8:0, 0:0, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<0, 1>>(coord))
        == "[ 6:0, 7:0, 8:0, 0:1, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<0, 2>>(coord))
        == "[ 6:0, 7:0, 8:0, 0:2, 0:0 ]");
    CHECK(tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<1>>(coord)) == "[ 6:0, 7:0, 8:1, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<2, 0>>(coord))
        == "[ 6:0, 7:0, 8:2, 0:0, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<2, 1>>(coord))
        == "[ 6:0, 7:0, 8:2, 0:1, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<2, 2>>(coord))
        == "[ 6:0, 7:0, 8:2, 0:2, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<3, 0>>(coord))
        == "[ 6:0, 7:0, 8:3, 0:0, 0:0 ]");
    CHECK(
        tree::treeCoordToString(tree::createTreeCoord<llama::RecordCoord<3, 1>>(coord))
        == "[ 6:0, 7:0, 8:3, 0:1, 0:0 ]");
}
