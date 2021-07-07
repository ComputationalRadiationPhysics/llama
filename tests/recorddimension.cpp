#include "common.h"

#include <array>
#include <atomic>
#include <catch2/catch.hpp>
#include <complex>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <vector>

namespace
{
    struct Tag
    {
    };
} // namespace

TEST_CASE("recorddim.record_with_int")
{
    using RecordDim = llama::Record<llama::Field<Tag, int>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    int& e = view(0u)(Tag{});
    e = 0;
}

TEST_CASE("recorddim.record_with_int[3]")
{
    using namespace llama::literals;

    using RecordDim = llama::Record<llama::Field<Tag, int[3]>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    int& e0 = view(0u)(Tag{})(0_RC);
    int& e1 = view(0u)(Tag{})(1_RC);
    int& e2 = view(0u)(Tag{})(2_RC);
}

// TEST_CASE("recorddim.record_with_std::complex<float>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::complex<float>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    std::complex<float>& e = view(0u)(Tag{});
//    e = {2, 3};
//}

TEST_CASE("recorddim.record_with_std::array<float, 4>")
{
    using RecordDim = llama::Record<llama::Field<Tag, std::array<float, 4>>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    std::array<float, 4>& e = view(0u)(Tag{});
    e = {2, 3, 4, 5};
}

// TEST_CASE("recorddim.record_with_std::vector<float>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::vector<float>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    std::vector<float>& e = view(0u)(Tag{});
//    // e = {2, 3, 4, 5}; // FIXME: LLAMA memory is uninitialized
//}

// TEST_CASE("recorddim.record_with_std::atomic<int>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::atomic<int>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    std::atomic<int>& e = view(0u)(Tag{});
//    // e++; // FIXME: LLAMA memory is uninitialized
//}

TEST_CASE("recorddim.record_with_noncopyable")
{
    struct Element
    {
        Element() = default;
        Element(const Element&) = delete;
        auto operator=(const Element&) -> Element& = delete;
        Element(Element&&) noexcept = default;
        auto operator=(Element&&) noexcept -> Element& = default;
        ~Element() = default;

        int value;
    };

    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    Element& e = view(0u)(Tag{});
    e.value = 0;
}

TEST_CASE("recorddim.record_with_nonmoveable")
{
    struct Element
    {
        Element() = default;
        Element(const Element&) = delete;
        auto operator=(const Element&) -> Element& = delete;
        Element(Element&&) noexcept = delete;
        auto operator=(Element&&) noexcept -> Element& = delete;
        ~Element() = default;

        int value;
    };

    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    Element& e = view(0u)(Tag{});
    e.value = 0;
}

// TEST_CASE("recorddim.record_with_nondefaultconstructible")
//{
//    struct Element
//    {
//        Element() = delete;
//        int value;
//    };
//
//    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    Element& e = view(0u)(Tag{});
//    e.value = 0;
//}

// TEST_CASE("recorddim.record_with_nontrivial_ctor")
//{
//    struct Element
//    {
//        int value = 42;
//    };
//
//    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    Element& e = view(0u)(Tag{});
//    // CHECK(e.value == 42); // FIXME: LLAMA memory is uninitialized
//}

// namespace
//{
//    struct UniqueInt
//    {
//        int value = counter++;
//
//        explicit operator int() const
//        {
//            return value;
//        }
//
//    private:
//        inline static int counter = 0;
//    };
//} // namespace
//
// TEST_CASE("recorddim.record_with_nontrivial_ctor2")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, UniqueInt>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    // FIXME: LLAMA memory is uninitialized
//    // CHECK(view(ArrayDims{0})(Tag{}) == 0);
//    // CHECK(view(ArrayDims{1})(Tag{}) == 1);
//    // CHECK(view(ArrayDims{2})(Tag{}) == 2);
//    // CHECK(view(ArrayDims{15})(Tag{}) == 15);
//}

TEST_CASE("recorddim.int")
{
    using RecordDim = int;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    STATIC_REQUIRE(std::is_same_v<decltype(view(0u)), int&>);
    view(0u) = 42;
    CHECK(view(0u) == 42);

    STATIC_REQUIRE(std::is_same_v<decltype(view[0u]), int&>);
    view[0u] = 42;
    CHECK(view[0u] == 42);

    STATIC_REQUIRE(std::is_same_v<decltype(view[llama::ArrayDims{0}]), int&>);
    view[llama::ArrayDims{0}] = 42;
    CHECK(view[llama::ArrayDims{0}] == 42);
}

TEST_CASE("recorddim.int[3]")
{
    using namespace llama::literals;

    using RecordDim = int[3];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC) = 42;
    view(0u)(1_RC) = 43;
    view(0u)(2_RC) = 44;
}

// if we lift the array size higher, we hit the limit on template instantiations
TEST_CASE("recorddim.int[200]")
{
    using namespace llama::literals;

    using RecordDim = int[200];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC) = 42;
    view(0u)(199_RC) = 43;
}

TEST_CASE("recorddim.int[3][2]")
{
    using namespace llama::literals;

    using RecordDim = int[3][2];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC)(0_RC) = 42;
    view(0u)(0_RC)(1_RC) = 43;
    view(0u)(1_RC)(0_RC) = 44;
    view(0u)(1_RC)(1_RC) = 45;
    view(0u)(2_RC)(0_RC) = 46;
    view(0u)(2_RC)(1_RC) = 47;
}

TEST_CASE("recorddim.int[1][1][1][1][1][1][1][1][1][1]")
{
    using namespace llama::literals;

    using RecordDim = int[1][1][1][1][1][1][1][1][1][1];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC)(0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) = 42;
}

// clang-format off
struct A1{};
struct A2{};
struct A3{};

using Arrays = llama::Record<
    llama::Field<A1, int[3]>,
    llama::Field<A2, llama::Record<
        llama::Field<Tag, float>
    >[3]>,
    llama::Field<A3, int[2][2]>
>;
// clang-format on

TEST_CASE("recorddim.record_with_arrays")
{
    using namespace llama::literals;

    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{1}, Arrays{}});
    view(0u)(A1{}, 0_RC);
    view(0u)(A1{}, 1_RC);
    view(0u)(A1{}, 2_RC);
    view(0u)(A2{}, 0_RC, Tag{});
    view(0u)(A2{}, 1_RC, Tag{});
    view(0u)(A2{}, 2_RC, Tag{});
    view(0u)(A3{}, 0_RC, 0_RC);
    view(0u)(A3{}, 0_RC, 1_RC);
    view(0u)(A3{}, 1_RC, 0_RC);
    view(0u)(A3{}, 1_RC, 1_RC);
}

TEST_CASE("dynamic array")
{
    struct Tag
    {
    };
    using RecordDim = llama::Record<llama::Field<Tag, int[]>>;
    auto mapping
        = llama::mapping::OffsetTable<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{2}, llama::ArrayDims{5}};
    auto view = allocView(mapping);

    view(0)(llama::EndOffset<Tag>{}) = 3;
    view(1)(llama::EndOffset<Tag>{}) = 5;

    CHECK(view(0)(llama::Size<Tag>{}) == 3);
    int& e0 = view(0)(Tag{})(0);
    int& e1 = view(0)(Tag{})(1);
    int& e2 = view(0)(Tag{})(2);
    CHECK(view(1)(llama::Size<Tag>{}) == 2);
    int& e3 = view(1)(Tag{})(0);
    int& e4 = view(1)(Tag{})(1);

    e0 = 1;
    e1 = 2;
    e2 = 3;
    e3 = 4;
    e4 = 5;
    CHECK(e0 == 1);
    CHECK(e1 == 2);
    CHECK(e2 == 3);
    CHECK(e3 == 4);
    CHECK(e4 == 5);
}

namespace
{
    // clang-format off
    struct run {};
    struct luminosityBlock {};
    struct Electrons {};
    struct Muons {};
    struct Eta{};
    struct Mass{};
    struct Phi{};

    using Electron = llama::Record<
        llama::Field<Eta, float>,
        llama::Field<Mass, float>,
        llama::Field<Phi, float>
    >;
    using Muon = llama::Record<
        llama::Field<Eta, float>,
        llama::Field<Mass, float>,
        llama::Field<Phi, float>
    >;
    using Event = llama::Record<
        llama::Field<run, std::int32_t>,
        llama::Field<luminosityBlock, std::int32_t>,
        llama::Field<Electrons, Electron[]>,
        llama::Field<Muons, Muon[]>
    >;
    // clang-format on
} // namespace

TEST_CASE("edm")
{
    // 3 events with 5 electrons and 4 muons
    auto mapping = llama::mapping::OffsetTable<llama::ArrayDims<1>, Event>{
        llama::ArrayDims{3},
        llama::ArrayDims{5},
        llama::ArrayDims{4}};
    auto view = llama::allocView(mapping);

    // setup offset table
    view(0)(llama::EndOffset<Electrons>{}) = 3;
    view(1)(llama::EndOffset<Electrons>{}) = 3;
    view(2)(llama::EndOffset<Electrons>{}) = 5;

    view(0)(llama::EndOffset<Muons>{}) = 0;
    view(1)(llama::EndOffset<Muons>{}) = 3;
    view(2)(llama::EndOffset<Muons>{}) = 4;

    // fill with values
    int value = 1;
    for (auto i = 0; i < 3; i++)
    {
        auto event = view(i);
        event(run{}) = value++;
        event(luminosityBlock{}) = value++;
        for (auto j = 0; j < event(llama::Size<Electrons>{}); j++)
        {
            auto electron = event(Electrons{})(j);
            electron(Eta{}) = value++;
            electron(Mass{}) = value++;
            electron(Phi{}) = value++;
        }
        for (auto j = 0; j < event(llama::Size<Muons>{}); j++)
        {
            auto muon = event(Muons{})(j);
            muon(Eta{}) = value++;
            muon(Mass{}) = value++;
            muon(Phi{}) = value++;
        }
    }

    // check all values
    value = 1;
    CHECK(view(0)(run{}) == value++);
    CHECK(view(0)(luminosityBlock{}) == value++);
    CHECK(view(0)(llama::EndOffset<Electrons>{}) == 3);
    CHECK(view(0)(llama::Size<Electrons>{}) == 3);
    CHECK(view(0)(Electrons{})(0) (Eta{}) == value++);
    CHECK(view(0)(Electrons{})(0) (Mass{}) == value++);
    CHECK(view(0)(Electrons{})(0) (Phi{}) == value++);
    CHECK(view(0)(Electrons{})(1) (Eta{}) == value++);
    CHECK(view(0)(Electrons{})(1) (Mass{}) == value++);
    CHECK(view(0)(Electrons{})(1) (Phi{}) == value++);
    CHECK(view(0)(Electrons{})(2) (Eta{}) == value++);
    CHECK(view(0)(Electrons{})(2) (Mass{}) == value++);
    CHECK(view(0)(Electrons{})(2) (Phi{}) == value++);
    CHECK(view(0)(llama::EndOffset<Muons>{}) == 0);
    CHECK(view(0)(llama::Size<Muons>{}) == 0);

    CHECK(view(1)(run{}) == value++);
    CHECK(view(1)(luminosityBlock{}) == value++);
    CHECK(view(1)(llama::EndOffset<Electrons>{}) == 3);
    CHECK(view(1)(llama::Size<Electrons>{}) == 0);
    CHECK(view(1)(llama::EndOffset<Muons>{}) == 3);
    CHECK(view(1)(llama::Size<Muons>{}) == 3);
    CHECK(view(1)(Muons{})(0) (Eta{}) == value++);
    CHECK(view(1)(Muons{})(0) (Mass{}) == value++);
    CHECK(view(1)(Muons{})(0) (Phi{}) == value++);
    CHECK(view(1)(Muons{})(1) (Eta{}) == value++);
    CHECK(view(1)(Muons{})(1) (Mass{}) == value++);
    CHECK(view(1)(Muons{})(1) (Phi{}) == value++);
    CHECK(view(1)(Muons{})(2) (Eta{}) == value++);
    CHECK(view(1)(Muons{})(2) (Mass{}) == value++);
    CHECK(view(1)(Muons{})(2) (Phi{}) == value++);

    CHECK(view(2)(run{}) == value++);
    CHECK(view(2)(luminosityBlock{}) == value++);
    CHECK(view(2)(llama::EndOffset<Electrons>{}) == 5);
    CHECK(view(2)(llama::Size<Electrons>{}) == 2);
    CHECK(view(2)(Electrons{})(0) (Eta{}) == value++);
    CHECK(view(2)(Electrons{})(0) (Mass{}) == value++);
    CHECK(view(2)(Electrons{})(0) (Phi{}) == value++);
    CHECK(view(2)(Electrons{})(1) (Eta{}) == value++);
    CHECK(view(2)(Electrons{})(1) (Mass{}) == value++);
    CHECK(view(2)(Electrons{})(1) (Phi{}) == value++);
    CHECK(view(2)(llama::EndOffset<Muons>{}) == 4);
    CHECK(view(2)(llama::Size<Muons>{}) == 1);
    CHECK(view(2)(Muons{})(0) (Eta{}) == value++);
    CHECK(view(2)(Muons{})(0) (Mass{}) == value++);
    CHECK(view(2)(Muons{})(0) (Phi{}) == value++);
}

TEST_CASE("dump.edm.AlignedAoS")
{
    auto mapping = llama::mapping::OffsetTable<llama::ArrayDims<1>, Event>{
        llama::ArrayDims{30},
        llama::ArrayDims{50},
        llama::ArrayDims{40}};
    std::ofstream{"dump.edm.AlignedAoS.svg"} << llama::toSvg(mapping);
    std::ofstream{"dump.edm.AlignedAoS.html"} << llama::toHtml(mapping);
}

TEST_CASE("dump.edm.MultiBlobSoA")
{
    auto mapping = llama::mapping::
        OffsetTable<llama::ArrayDims<1>, Event, llama::mapping::MappingList<llama::mapping::PreconfiguredSoA<>::type>>{
            llama::ArrayDims{30},
            llama::ArrayDims{50},
            llama::ArrayDims{40}};
    std::ofstream{"dump.edm.MultiBlobSoA.svg"} << llama::toSvg(mapping);
    std::ofstream{"dump.edm.MultiBlobSoA.html"} << llama::toHtml(mapping);
}

TEST_CASE("dump.edm.AlignedAoS_MultiBlobSoA")
{
    auto mapping = llama::mapping::OffsetTable<
        llama::ArrayDims<1>,
        Event,
        llama::mapping::MappingList<
            llama::mapping::PreconfiguredAoS<>::type,
            llama::mapping::PreconfiguredSoA<>::type,
            llama::mapping::PreconfiguredSoA<>::type>>{
        llama::ArrayDims{30},
        llama::ArrayDims{50},
        llama::ArrayDims{40}};
    std::ofstream{"dump.edm.AlignedAoS_MultiBlobSoA.svg"} << llama::toSvg(mapping);
    std::ofstream{"dump.edm.AlignedAoS_MultiBlobSoA.html"} << llama::toHtml(mapping);
}

TEST_CASE("dump.edm.Split_AlignedAoS_MultiBlobSoA")
{
    auto mapping = llama::mapping::OffsetTable<
        llama::ArrayDims<1>,
        Event,
        llama::mapping::MappingList<
            llama::mapping::PreconfiguredSplit<
                llama::RecordCoord<2>,
                llama::mapping::PreconfiguredAoS<>::type,
                llama::mapping::PreconfiguredAoS<>::type,
                true>::type,
            llama::mapping::PreconfiguredSoA<>::type,
            llama::mapping::PreconfiguredSoA<>::type>>{
        llama::ArrayDims{30},
        llama::ArrayDims{50},
        llama::ArrayDims{40}};
    std::ofstream{"dump.edm.Split_AlignedAoS_MultiBlobSoA.svg"} << llama::toSvg(mapping);
    std::ofstream{"dump.edm.Split_AlignedAoS_MultiBlobSoA.html"} << llama::toHtml(mapping);
}
