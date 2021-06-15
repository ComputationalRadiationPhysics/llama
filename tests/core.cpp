#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("prettyPrintType")
{
    auto str = prettyPrintType<Particle>();
#ifdef _WIN32
    replace_all(str, "__int64", "long");
#endif
    const auto* const ref = R"(llama::Record<
    llama::Field<
        tag::Pos,
        llama::Record<
            llama::Field<
                tag::X,
                double
            >,
            llama::Field<
                tag::Y,
                double
            >,
            llama::Field<
                tag::Z,
                double
            >
        >
    >,
    llama::Field<
        tag::Mass,
        float
    >,
    llama::Field<
        tag::Vel,
        llama::Record<
            llama::Field<
                tag::X,
                double
            >,
            llama::Field<
                tag::Y,
                double
            >,
            llama::Field<
                tag::Z,
                double
            >
        >
    >,
    llama::Field<
        tag::Flags,
        bool [4]
    >
>)";
    CHECK(str == ref);
}

TEST_CASE("sizeOf")
{
    STATIC_REQUIRE(llama::sizeOf<float> == 4);
    STATIC_REQUIRE(llama::sizeOf<Vec3D> == 24);
    STATIC_REQUIRE(llama::sizeOf<Vec2F> == 8);
    STATIC_REQUIRE(llama::sizeOf<Particle> == 56);
}

TEST_CASE("sizeOf.Align")
{
    STATIC_REQUIRE(llama::sizeOf<float, true> == 4);
    STATIC_REQUIRE(llama::sizeOf<Vec3D, true> == 24);
    STATIC_REQUIRE(llama::sizeOf<Vec2F, true> == 8);
    STATIC_REQUIRE(llama::sizeOf<Particle, true> == 64);
}

TEST_CASE("offsetOf")
{
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 0>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 1>> == 8);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 2>> == 16);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<1>> == 24);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2>> == 28);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 0>> == 28);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 1>> == 36);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 2>> == 44);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3>> == 52);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 0>> == 52);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 1>> == 53);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 2>> == 54);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 3>> == 55);
}

TEST_CASE("offsetOf.Align")
{
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 1>, true> == 8);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 2>, true> == 16);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<1>, true> == 24);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2>, true> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 0>, true> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 1>, true> == 40);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 2>, true> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3>, true> == 56);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 0>, true> == 56);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 1>, true> == 57);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 2>, true> == 58);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 3>, true> == 59);
}

namespace
{
    // clang-format off
    using Other = llama::Record<
        llama::Field<tag::Pos, llama::Record<
            llama::Field<tag::Z, float>,
            llama::Field<tag::Y, float>
        >>
    >;
    // clang-format on
} // namespace

TEST_CASE("fieldCount")
{
    STATIC_REQUIRE(llama::flatFieldCount<int> == 1);
    STATIC_REQUIRE(llama::flatFieldCount<Vec3D> == 3);
    STATIC_REQUIRE(llama::flatFieldCount<Particle> == 11);
    STATIC_REQUIRE(llama::flatFieldCount<Other> == 2);
}

TEST_CASE("flatFieldCountBefore")
{
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<0, Particle> == 0);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<1, Particle> == 3);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<2, Particle> == 4);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<3, Particle> == 7);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<4, Particle> == 11);
}

template <int i>
struct S;

TEST_CASE("alignment")
{
    using RD = llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, bool>,
        llama::Field<tag::Mass, std::uint16_t>>;

    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<>, false> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<0>, false> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<1>, false> == 4);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<2>, false> == 12);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<3>, false> == 13);
    STATIC_REQUIRE(llama::sizeOf<RD, false> == 15);

    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<1>, true> == 8); // aligned
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<2>, true> == 16);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<3>, true> == 18); // aligned
    STATIC_REQUIRE(llama::sizeOf<RD, true> == 24);
}

TEST_CASE("GetCoordFromTags")
{
    using namespace llama::literals;
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle                                   >, llama::RecordCoord<    >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos                         >, llama::RecordCoord<0   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::X                 >, llama::RecordCoord<0, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Y                 >, llama::RecordCoord<0, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Z                 >, llama::RecordCoord<0, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Mass                        >, llama::RecordCoord<1   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::X                 >, llama::RecordCoord<2, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Y                 >, llama::RecordCoord<2, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Z                 >, llama::RecordCoord<2, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags                       >, llama::RecordCoord<3   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<0>>, llama::RecordCoord<3, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<1>>, llama::RecordCoord<3, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<2>>, llama::RecordCoord<3, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<3>>, llama::RecordCoord<3, 3>>);
    // clang-format on
}

TEST_CASE("GetTags")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0, 0>>, boost::mp11::mp_list<llama::NoName, tag::Pos, tag::X >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0   >>, boost::mp11::mp_list<llama::NoName, tag::Pos         >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<    >>, boost::mp11::mp_list<llama::NoName                   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<2, 1>>, boost::mp11::mp_list<llama::NoName, tag::Vel, tag::Y >>);
    // clang-format on
}

TEST_CASE("GetTag")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<0, 0>>, tag::X       >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<0   >>, tag::Pos     >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<    >>, llama::NoName>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<2, 1>>, tag::Y       >);
    // clang-format on
}

TEST_CASE("hasSameTags")
{
    using PosRecord = llama::GetType<Particle, llama::GetCoordFromTags<Particle, tag::Pos>>;
    using VelRecord = llama::GetType<Particle, llama::GetCoordFromTags<Particle, tag::Vel>>;

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosRecord, // RD A
            llama::RecordCoord<0>, // Local A
            VelRecord, // RD B
            llama::RecordCoord<1> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosRecord, // RD A
            llama::RecordCoord<0>, // Local A
            VelRecord, // RD B
            llama::RecordCoord<0> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<0, 0>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<0, 2>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<3, 0>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            > == false);
}

TEST_CASE("FlatRecordDim")
{
    STATIC_REQUIRE(
        std::is_same_v<
            llama::FlatRecordDim<Particle>,
            boost::mp11::mp_list<double, double, double, float, double, double, double, bool, bool, bool, bool>>);
}

TEST_CASE("flatRecordCoord")
{
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<>> == 0);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0>> == 0);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0, 0>> == 0);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0, 1>> == 1);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0, 2>> == 2);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<1>> == 3);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2>> == 4);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2, 0>> == 4);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2, 1>> == 5);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2, 2>> == 6);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3>> == 7);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 0>> == 7);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 1>> == 8);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 2>> == 9);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 3>> == 10);
}