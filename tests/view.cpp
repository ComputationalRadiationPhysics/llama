#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Value {};
} // namespace tag

using RecordDim = llama::Record<
    llama::Field<tag::Value, int>
>;
// clang-format on

TEST_CASE("view.default-ctor")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{16, 16};

    [[maybe_unused]] llama::View<llama::mapping::SoA<ArrayDims, RecordDim>, std::byte*> view1{};
    [[maybe_unused]] llama::View<llama::mapping::AoS<ArrayDims, RecordDim>, std::byte*> view2{};
    [[maybe_unused]] llama::View<llama::mapping::One<ArrayDims, RecordDim>, std::byte*> view3{};
    [[maybe_unused]] llama::View<llama::mapping::tree::Mapping<ArrayDims, RecordDim, llama::Tuple<>>, std::byte*> view4;
}

TEST_CASE("view.move")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    auto view1 = allocView(Mapping(view_size));

    decltype(view1) view2;
    view1({3, 3}) = 1;
    view2 = std::move(view1);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.swap")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    auto view1 = allocView(Mapping(view_size));
    auto view2 = allocView(Mapping(view_size));

    view1({3, 3}) = 1;
    view2({3, 3}) = 2;

    std::swap(view1, view2);

    CHECK(view1({3, 3}) == 2);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.allocator.Vector")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    auto view = allocView(Mapping(view_size), llama::bloballoc::Vector{});

    for (auto i : llama::ArrayDimsIndexRange{view_size})
        view(i) = 42;
}

TEST_CASE("view.allocator.SharedPtr")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    auto view = allocView(Mapping(view_size), llama::bloballoc::SharedPtr{});

    for (auto i : llama::ArrayDimsIndexRange{view_size})
        view(i) = 42;
}

TEST_CASE("view.allocator.stack")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    auto view = allocView(Mapping(view_size), llama::bloballoc::Stack<16 * 16 * llama::sizeOf<RecordDim>>{});

    for (auto i : llama::ArrayDimsIndexRange{view_size})
        view(i) = 42;
}

TEST_CASE("view.non-memory-owning")
{
    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims array_dims{256};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    Mapping mapping{array_dims};

    std::vector<std::byte> storage(mapping.blobSize(0));
    auto view = llama::View<Mapping, std::byte*>{mapping, {storage.data()}};

    for (auto i = 0u; i < 256u; i++)
    {
        auto* v = reinterpret_cast<std::byte*>(&view(i)(tag::Value{}));
        CHECK(&storage.front() <= v);
        CHECK(v <= &storage.back());
    }
}

TEST_CASE("view.access")
{
    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    zeroStorage(view);

    auto l = [](auto& view)
    {
        const ArrayDims pos{0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const auto& x = view(pos)(llama::RecordCoord<0, 0>{});
        CHECK(&x == &view(pos)(llama::RecordCoord<0>{})(llama::RecordCoord<0>{}));
        CHECK(&x == &view(pos)(llama::RecordCoord<0>{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{})(llama::RecordCoord<0>{}));
        CHECK(&x == &view(pos)(tag::Pos{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{}, tag::X{}));

        // also test arrays
        using namespace llama::literals;
        const bool& o0 = view(pos)(tag::Flags{})(llama::RecordCoord<0>{});
        CHECK(&o0 == &view(pos)(tag::Flags{})(0_RC));
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("view.assign-one-record")
{
    using namespace llama::literals;

    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    llama::One<Particle> record;
    record(tag::Pos{}, tag::X{}) = 14.0f;
    record(tag::Pos{}, tag::Y{}) = 15.0f;
    record(tag::Pos{}, tag::Z{}) = 16.0f;
    record(tag::Vel{}) = 0;
    record(tag::Mass{}) = 500.0f;
    record(tag::Flags{})(0_RC) = true;
    record(tag::Flags{})(1_RC) = false;
    record(tag::Flags{})(2_RC) = true;
    record(tag::Flags{})(3_RC) = false;

    view({3, 4}) = record;

    CHECK(record(tag::Pos{}, tag::X{}) == 14.0f);
    CHECK(record(tag::Pos{}, tag::Y{}) == 15.0f);
    CHECK(record(tag::Pos{}, tag::Z{}) == 16.0f);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Mass{}) == 500.0f);
    CHECK(record(tag::Flags{})(0_RC) == true);
    CHECK(record(tag::Flags{})(1_RC) == false);
    CHECK(record(tag::Flags{})(2_RC) == true);
    CHECK(record(tag::Flags{})(3_RC) == false);
}

TEST_CASE("view.addresses")
{
    using namespace llama::literals;

    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    const ArrayDims pos{0, 0};
    auto& x = view(pos)(tag::Pos{}, tag::X{});
    auto& y = view(pos)(tag::Pos{}, tag::Y{});
    auto& z = view(pos)(tag::Pos{}, tag::Z{});
    auto& w = view(pos)(tag::Mass{});
    auto& mx = view(pos)(tag::Vel{}, tag::X{});
    auto& my = view(pos)(tag::Vel{}, tag::Y{});
    auto& mz = view(pos)(tag::Vel{}, tag::Z{});
    auto& o0 = view(pos)(tag::Flags{})(0_RC);
    auto& o1 = view(pos)(tag::Flags{})(1_RC);
    auto& o2 = view(pos)(tag::Flags{})(2_RC);
    auto& o3 = view(pos)(tag::Flags{})(3_RC);

    CHECK(reinterpret_cast<std::byte*>(&y) - reinterpret_cast<std::byte*>(&x) == 2048);
    CHECK(reinterpret_cast<std::byte*>(&z) - reinterpret_cast<std::byte*>(&x) == 4096);
    CHECK(reinterpret_cast<std::byte*>(&mx) - reinterpret_cast<std::byte*>(&x) == 7168);
    CHECK(reinterpret_cast<std::byte*>(&my) - reinterpret_cast<std::byte*>(&x) == 9216);
    CHECK(reinterpret_cast<std::byte*>(&mz) - reinterpret_cast<std::byte*>(&x) == 11264);
    CHECK(reinterpret_cast<std::byte*>(&w) - reinterpret_cast<std::byte*>(&x) == 6144);
    CHECK(reinterpret_cast<std::byte*>(&o0) - reinterpret_cast<std::byte*>(&x) == 13312);
    CHECK(reinterpret_cast<std::byte*>(&o1) - reinterpret_cast<std::byte*>(&x) == 13568);
    CHECK(reinterpret_cast<std::byte*>(&o2) - reinterpret_cast<std::byte*>(&x) == 13824);
    CHECK(reinterpret_cast<std::byte*>(&o3) - reinterpret_cast<std::byte*>(&x) == 14080);
}

template <typename TVirtualRecord>
struct SetZeroFunctor
{
    template <typename TCoord>
    void operator()(TCoord coord)
    {
        vd(coord) = 0;
    }
    TVirtualRecord vd;
};

TEST_CASE("view.iteration-and-access")
{
    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    zeroStorage(view);

    for (size_t x = 0; x < array_dims[0]; ++x)
        for (size_t y = 0; y < array_dims[1]; ++y)
        {
            SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
            llama::forEachLeaf<Particle>(szf, llama::RecordCoord<0, 0>{});
            llama::forEachLeaf<Particle>(szf, tag::Vel{});
            view({x, y}) = double(x + y) / double(array_dims[0] + array_dims[1]);
        }

    double sum = 0.0;
    for (size_t x = 0; x < array_dims[0]; ++x)
        for (size_t y = 0; y < array_dims[1]; ++y)
            sum += view(x, y)(llama::RecordCoord<2, 0>{});
    CHECK(sum == 120.0);
}

TEST_CASE("view.record-access")
{
    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    zeroStorage(view);

    for (size_t x = 0; x < array_dims[0]; ++x)
        for (size_t y = 0; y < array_dims[1]; ++y)
        {
            auto record = view(x, y);
            record(tag::Pos(), tag::X()) += record(llama::RecordCoord<2, 0>{});
            record(tag::Pos(), tag::Y()) += record(llama::RecordCoord<2, 1>{});
            record(tag::Pos(), tag::Z()) += record(llama::RecordCoord<1>());
            record(tag::Pos()) += record(tag::Vel());
        }

    double sum = 0.0;
    for (size_t x = 0; x < array_dims[0]; ++x)
        for (size_t y = 0; y < array_dims[1]; ++y)
            sum += view(x, y)(llama::RecordCoord<2, 0>{});

    CHECK(sum == 0.0);
}

TEST_CASE("view.indexing")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{16, 16}, Particle{}});
    view(0u, 0u)(tag::Mass{}) = 42.0f;

    using Integrals = boost::mp11::
        mp_list<char, unsigned char, signed char, short, unsigned short, int, unsigned int, long, unsigned long>;

    boost::mp11::mp_for_each<integrals>(
        [&](auto i)
        {
            boost::mp11::mp_for_each<integrals>(
                [&](auto j)
                {
                    const float& w = view(i, j)(tag::Mass{});
                    CHECK(w == 42.0f);
                });
        });

    llama::VirtualView virtual_view{view, {0, 0}};
    boost::mp11::mp_for_each<integrals>(
        [virtual_view](auto i)
        {
            boost::mp11::mp_for_each<integrals>(
                [virtual_view](auto j)
                {
                    const float& w = virtual_view(i, j)(tag::Mass{});
                    CHECK(w == 42.0f);
                });
        });
}
