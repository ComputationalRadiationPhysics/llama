#include <algorithm>
#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <numeric>

// clang-format off
namespace tag {
    struct X {};
    struct Y {};
    struct Z {};
} // namespace tag

using Position = llama::Record<
    llama::Field<tag::X, int>,
    llama::Field<tag::Y, int>,
    llama::Field<tag::Z, int>
>;
// clang-format on

TEST_CASE("iterator")
{
    auto test = [](auto array_dims)
    {
        using ArrayDims = decltype(array_dims);
        auto mapping = llama::mapping::AoS<ArrayDims, Position>{array_dims};
        auto view = llama::allocView(mapping);

        for (auto vd : view)
        {
            vd(tag::X{}) = 1;
            vd(tag::Y{}) = 2;
            vd(tag::Z{}) = 3;
        }
        std::transform(begin(view), end(view), begin(view), [](auto vd) { return vd * 2; });
        const auto& cview = std::as_const(view);
        const int sum_y
            = std::accumulate(begin(cview), end(cview), 0, [](int acc, auto vd) { return acc + vd(tag::Y{}); });
        CHECK(sum_y == 128);
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.std_copy")
{
    auto test = [](auto array_dims)
    {
        auto aos_view = llama::allocView(llama::mapping::AoS{array_dims, Position{}});
        auto soa_view = llama::allocView(llama::mapping::SoA{array_dims, Position{}});

        int i = 0;
        for (auto vd : aos_view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }
        std::copy(begin(aos_view), end(aos_view), begin(soa_view));
        i = 0;
        for (auto vd : soa_view)
        {
            CHECK(vd(tag::X{}) == ++i);
            CHECK(vd(tag::Y{}) == ++i);
            CHECK(vd(tag::Z{}) == ++i);
        }
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.transform_reduce")
{
    auto test = [](auto array_dims)
    {
        auto aos_view = llama::allocView(llama::mapping::AoS{array_dims, Position{}});
        auto soa_view = llama::allocView(llama::mapping::SoA{array_dims, Position{}});

        int i = 0;
        for (auto vd : aos_view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }
        for (auto vd : soa_view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }
        // returned type is a llama::One<Position>
        auto [sumX, sumY, sumZ]
            = std::transform_reduce(begin(aos_view), end(aos_view), begin(soa_view), llama::One<Position>{});

        CHECK(sumX == 242672);
        CHECK(sumY == 248816);
        CHECK(sumZ == 255024);
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.different_record_dim")
{
    struct Pos1
    {
    };
    struct Pos2
    {
    };
    using WrappedPos = llama::Record<llama::Field<Pos1, Position>, llama::Field<Pos2, Position>>;

    auto array_dims = llama::ArrayDims{32};
    auto aos_view = llama::allocView(llama::mapping::AoS{array_dims, WrappedPos{}});
    auto soa_view = llama::allocView(llama::mapping::SoA{array_dims, Position{}});

    int i = 0;
    for (auto vd : aos_view)
    {
        vd(Pos1{}, tag::X{}) = ++i;
        vd(Pos1{}, tag::Y{}) = ++i;
        vd(Pos1{}, tag::Z{}) = ++i;
    }
    std::transform(begin(aos_view), end(aos_view), begin(soa_view), [](auto wp) { return wp(Pos1{}) * 2; });

    i = 0;
    for (auto vd : soa_view)
    {
        CHECK(vd(tag::X{}) == ++i * 2);
        CHECK(vd(tag::Y{}) == ++i * 2);
        CHECK(vd(tag::Z{}) == ++i * 2);
    }
}

// TODO(bgruber): clang 10 and 11 fail to compile this currently with the issue described here:
// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
// let's try again with clang 12
// Intel LLVM compiler is also using the clang frontend
#if CAN_USE_RANGES
#    include <ranges>

TEST_CASE("ranges")
{
    auto test = [](auto arrayDims)
    {
        auto mapping = llama::mapping::AoS{arrayDims, Position{}};
        auto view = llama::allocView(mapping);

        STATIC_REQUIRE(std::ranges::range<decltype(view)>);

        int i = 0;
        for (auto vd : view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }

        std::vector<int> v;
        for (auto y : view | std::views::filter([](auto vd) { return vd(tag::X{}) % 10 == 0; })
                 | std::views::transform([](auto vd) { return vd(tag::Y{}); }) | std::views::take(2))
            v.push_back(y);
        CHECK(v == std::vector<int>{11, 41});
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}
#endif
