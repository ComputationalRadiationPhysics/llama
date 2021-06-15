#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

template <typename TVirtualRecord>
struct DoubleFunctor
{
    template <typename TCoord>
    void operator()(TCoord coord)
    {
        vd(coord) *= 2;
    }
    TVirtualRecord vd;
};

TEST_CASE("virtual view CTAD")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{10, 10};
    auto view = allocView(llama::mapping::SoA<ArrayDims, Particle>(view_size));

    llama::VirtualView virtual_view{view, {2, 4}};
}

TEST_CASE("fast virtual view")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{10, 10};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    auto view = allocView(Mapping(view_size));

    for (std::size_t x = 0; x < view_size[0]; ++x)
        for (std::size_t y = 0; y < view_size[1]; ++y)
            view(x, y) = x * y;

    llama::VirtualView<decltype(view)> virtual_view{view, {2, 4}};

    CHECK(virtual_view.offset == ArrayDims{2, 4});

    CHECK(view(virtual_view.offset)(tag::Pos(), tag::X()) == 8.0);
    CHECK(virtual_view({0, 0})(tag::Pos(), tag::X()) == 8.0);

    CHECK(view({virtual_view.offset[0] + 2, virtual_view.offset[1] + 3})(tag::Vel(), tag::Z()) == 28.0);
    CHECK(virtual_view({2, 3})(tag::Vel(), tag::Z()) == 28.0);
}

TEST_CASE("virtual view")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims view_size{32, 32};
    constexpr ArrayDims mini_size{8, 8};
    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    auto view = allocView(Mapping(view_size));

    for (std::size_t x = 0; x < view_size[0]; ++x)
        for (std::size_t y = 0; y < view_size[1]; ++y)
            view(x, y) = x * y;

    constexpr ArrayDims iterations{
        (view_size[0] + mini_size[0] - 1) / mini_size[0],
        (view_size[1] + mini_size[1] - 1) / mini_size[1]};

    for (std::size_t x = 0; x < iterations[0]; ++x)
        for (std::size_t y = 0; y < iterations[1]; ++y)
        {
            const ArrayDims valid_mini_size{
                (x < iterations[0] - 1) ? mini_size[0] : (view_size[0] - 1) % mini_size[0] + 1,
                (y < iterations[1] - 1) ? mini_size[1] : (view_size[1] - 1) % mini_size[1] + 1};

            llama::VirtualView<decltype(view)> virtual_view(view, {x * mini_size[0], y * mini_size[1]});

            using MiniMapping = llama::mapping::SoA<ArrayDims, Particle>;
            auto mini_view = allocView(
                MiniMapping(mini_size),
                llama::bloballoc::Stack<mini_size[0] * mini_size[1] * llama::sizeOf<Particle>>{});

            for (std::size_t a = 0; a < valid_mini_size[0]; ++a)
                for (std::size_t b = 0; b < valid_mini_size[1]; ++b)
                    mini_view(a, b) = virtual_view(a, b);

            for (std::size_t a = 0; a < valid_mini_size[0]; ++a)
                for (std::size_t b = 0; b < valid_mini_size[1]; ++b)
                {
                    DoubleFunctor<decltype(mini_view(a, b))> sqrt_f{mini_view(a, b)};
                    llama::forEachLeaf<Particle>(sqrt_f);
                }

            for (std::size_t a = 0; a < valid_mini_size[0]; ++a)
                for (std::size_t b = 0; b < valid_mini_size[1]; ++b)
                    virtual_view(a, b) = mini_view(a, b);
        }

    for (std::size_t x = 0; x < view_size[0]; ++x)
        for (std::size_t y = 0; y < view_size[1]; ++y)
            CHECK((view(x, y)) == x * y * 2);
}
