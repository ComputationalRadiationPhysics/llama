#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("ArrayDims.CTAD")
{
    llama::ArrayDims ad0{};
    llama::ArrayDims ad1{1};
    llama::ArrayDims ad2{1, 1};
    llama::ArrayDims ad3{1, 1, 1};

    STATIC_REQUIRE(decltype(ad0)::rank == 0);
    STATIC_REQUIRE(decltype(ad1)::rank == 1);
    STATIC_REQUIRE(decltype(ad2)::rank == 2);
    STATIC_REQUIRE(decltype(ad3)::rank == 3);
}

TEST_CASE("ArrayDims.dim0")
{
    using ArrayDims = llama::ArrayDims<0>;
    ArrayDims array_dims{};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    double& x1 = view(ArrayDims{})(tag::Pos{}, tag::X{});
    double& x2 = view()(tag::Pos{}, tag::X{});
}

TEST_CASE("ArrayDims.dim1")
{
    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims array_dims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    double& x = view(ArrayDims{0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDims.dim2")
{
    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    double& x = view(ArrayDims{0, 0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDims.dim3")
{
    using ArrayDims = llama::ArrayDims<3>;
    ArrayDims array_dims{16, 16, 16};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    double& x = view(ArrayDims{0, 0, 0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDims.dim10")
{
    using ArrayDims = llama::ArrayDims<10>;
    ArrayDims array_dims{2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    Mapping mapping{array_dims};
    auto view = allocView(mapping);

    double& x = view(ArrayDims{0, 0, 0, 0, 0, 0, 0, 0, 0, 0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDims.ctor")
{
    llama::ArrayDims<1> ad{};
    CHECK(ad[0] == 0);
}

TEST_CASE("ArrayDimsIndexIterator.concepts")
{
    STATIC_REQUIRE(std::is_same_v<
                   std::iterator_traits<llama::ArrayDimsIndexIterator<3>>::iterator_category,
                   std::random_access_iterator_tag>);

#ifdef __cpp_concepts
    STATIC_REQUIRE(std::random_access_iterator<llama::ArrayDimsIndexIterator<3>>);
#endif
}

TEST_CASE("ArrayDimsIndexIterator")
{
    llama::ArrayDimsIndexRange r{llama::ArrayDims<2>{3, 3}};

    llama::ArrayDimsIndexIterator it = std::begin(r);
    CHECK(*it == llama::ArrayDims{0, 0});
    it++;
    CHECK(*it == llama::ArrayDims{0, 1});
    ++it;
    CHECK(*it == llama::ArrayDims{0, 2});
    --it;
    CHECK(*it == llama::ArrayDims{0, 1});
    it--;
    CHECK(*it == llama::ArrayDims{0, 0});

    it = std::begin(r);
    it += 2;
    CHECK(*it == llama::ArrayDims{0, 2});
    it += 2;
    CHECK(*it == llama::ArrayDims{1, 1});
    it -= 2;
    CHECK(*it == llama::ArrayDims{0, 2});
    it -= 2;
    CHECK(*it == llama::ArrayDims{0, 0});

    it = std::begin(r);
    CHECK(it[2] == llama::ArrayDims{0, 2});
    CHECK(it[4] == llama::ArrayDims{1, 1});
    CHECK(it[8] == llama::ArrayDims{2, 2});

    it = std::begin(r);
    CHECK(*(it + 8) == llama::ArrayDims{2, 2});
    CHECK(*(8 + it) == llama::ArrayDims{2, 2});

    it += 8;
    CHECK(*it == llama::ArrayDims{2, 2});
    CHECK(*(it - 8) == llama::ArrayDims{0, 0});
    it -= 8;
    CHECK(*it == llama::ArrayDims{0, 0});

    CHECK(std::end(r) - std::begin(r) == 9);
    CHECK(std::begin(r) - std::end(r) == -9);

    it = std::begin(r) + 4;
    CHECK(*it == llama::ArrayDims{1, 1});
    CHECK(std::end(r) - it == 5);
    CHECK(it - std::begin(r) == 4);

    it = std::begin(r);
    CHECK(it == it);
    CHECK(it != it + 1);
    CHECK(it < it + 1);
}


TEST_CASE("ArrayDimsIndexIterator.operator+=")
{
    std::vector<llama::ArrayDims<2>> coords;
    llama::ArrayDimsIndexRange r{llama::ArrayDims<2>{3, 4}};
    for (auto it = r.begin(); it != r.end(); it += 2)
        coords.push_back(*it);

    CHECK(coords == std::vector<llama::ArrayDims<2>>{{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 2}});
}

TEST_CASE("ArrayDimsIndexIterator.constexpr")
{
    constexpr auto r = [&]() constexpr
    {
        bool b = true;
        llama::ArrayDimsIndexIterator it = std::begin(llama::ArrayDimsIndexRange{llama::ArrayDims<2>{3, 3}});
        b &= *it == llama::ArrayDims{0, 0};
        it++;
        b &= *it == llama::ArrayDims{0, 1};
        ++it;
        b &= *it == llama::ArrayDims{0, 2};
        return b;
    }
    ();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayDimsIndexRange1D")
{
    llama::ArrayDims<1> ad{3};

    std::vector<llama::ArrayDims<1>> coords;
    for (auto coord : llama::ArrayDimsIndexRange{ad})
        coords.push_back(coord);

    CHECK(coords == std::vector<llama::ArrayDims<1>>{{0}, {1}, {2}});
}

TEST_CASE("ArrayDimsIndexRange2D")
{
    llama::ArrayDims<2> ad{3, 3};

    std::vector<llama::ArrayDims<2>> coords;
    for (auto coord : llama::ArrayDimsIndexRange{ad})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDims<2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("ArrayDimsIndexRange3D")
{
    llama::ArrayDims<3> ad{3, 3, 3};

    std::vector<llama::ArrayDims<3>> coords;
    for (auto coord : llama::ArrayDimsIndexRange{ad})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDims<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

#if CAN_USE_RANGES
#    include <ranges>

TEST_CASE("ArrayDimsIndexRange.concepts")
{
    STATIC_REQUIRE(std::ranges::range<llama::ArrayDimsIndexRange<3>>);
    STATIC_REQUIRE(std::ranges::random_access_range<llama::ArrayDimsIndexRange<3>>);
    // STATIC_REQUIRE(std::ranges::view<llama::ArrayDimsIndexRange<3>>);
}

TEST_CASE("ArrayDimsIndexRange3D.reverse")
{
    llama::ArrayDims<3> ad{3, 3, 3};

    std::vector<llama::ArrayDims<3>> coords;
    for (auto coord : llama::ArrayDimsIndexRange{ad} | std::views::reverse)
        coords.push_back(coord);

    CHECK(coords == std::vector<llama::ArrayDims<3>>{{{2, 2, 2}, {2, 2, 1}, {2, 2, 0}, {2, 1, 2}, {2, 1, 1}, {2, 1, 0},
                                                      {2, 0, 2}, {2, 0, 1}, {2, 0, 0}, {1, 2, 2}, {1, 2, 1}, {1, 2, 0},
                                                      {1, 1, 2}, {1, 1, 1}, {1, 1, 0}, {1, 0, 2}, {1, 0, 1}, {1, 0, 0},
                                                      {0, 2, 2}, {0, 2, 1}, {0, 2, 0}, {0, 1, 2}, {0, 1, 1}, {0, 1, 0},
                                                      {0, 0, 2}, {0, 0, 1}, {0, 0, 0}}});
}
#endif

TEST_CASE("ArrayDimsIndexRange1D.constexpr")
{
    constexpr auto r = []() constexpr
    {
        llama::ArrayDims<1> ad{3};
        int i = 0;
        for (auto coord : llama::ArrayDimsIndexRange{ad})
        {
            if (i == 0 && coord != llama::ArrayDims<1>{0})
                return false;
            if (i == 1 && coord != llama::ArrayDims<1>{1})
                return false;
            if (i == 2 && coord != llama::ArrayDims<1>{2})
                return false;
            i++;
        }

        return true;
    }
    ();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayDimsIndexRange3D.destructering")
{
    llama::ArrayDims<3> ad{1, 1, 1};

    for (auto [x, y, z] : llama::ArrayDimsIndexRange{ad})
    {
        CHECK(x == 0);
        CHECK(y == 0);
        CHECK(z == 0);
    }
}

TEST_CASE("Morton")
{
    using ArrayDims = llama::ArrayDims<2>;
    ArrayDims array_dims{2, 3};

    llama::mapping::LinearizeArrayDimsMorton lin;
    CHECK(lin.size(ArrayDims{2, 3}) == 4 * 4);
    CHECK(lin.size(ArrayDims{2, 4}) == 4 * 4);
    CHECK(lin.size(ArrayDims{2, 5}) == 8 * 8);
    CHECK(lin.size(ArrayDims{8, 8}) == 8 * 8);

    CHECK(lin(ArrayDims{0, 0}, {}) == 0);
    CHECK(lin(ArrayDims{0, 1}, {}) == 1);
    CHECK(lin(ArrayDims{0, 2}, {}) == 4);
    CHECK(lin(ArrayDims{0, 3}, {}) == 5);
    CHECK(lin(ArrayDims{1, 0}, {}) == 2);
    CHECK(lin(ArrayDims{1, 1}, {}) == 3);
    CHECK(lin(ArrayDims{1, 2}, {}) == 6);
    CHECK(lin(ArrayDims{1, 3}, {}) == 7);
    CHECK(lin(ArrayDims{2, 0}, {}) == 8);
    CHECK(lin(ArrayDims{2, 1}, {}) == 9);
    CHECK(lin(ArrayDims{2, 2}, {}) == 12);
    CHECK(lin(ArrayDims{2, 3}, {}) == 13);
    CHECK(lin(ArrayDims{3, 0}, {}) == 10);
    CHECK(lin(ArrayDims{3, 1}, {}) == 11);
    CHECK(lin(ArrayDims{3, 2}, {}) == 14);
    CHECK(lin(ArrayDims{3, 3}, {}) == 15);
}

TEST_CASE("forEachADCoord_1D")
{
    llama::ArrayDims<1> ad_size{3};

    std::vector<llama::ArrayDims<1>> coords;
    llama::forEachADCoord(ad_size, [&](llama::ArrayDims<1> coord) { coords.push_back(coord); });

    CHECK(coords == std::vector<llama::ArrayDims<1>>{{0}, {1}, {2}});
}

TEST_CASE("forEachADCoord_2D")
{
    llama::ArrayDims<2> ad_size{3, 3};

    std::vector<llama::ArrayDims<2>> coords;
    llama::forEachADCoord(ad_size, [&](llama::ArrayDims<2> coord) { coords.push_back(coord); });

    CHECK(
        coords
        == std::vector<llama::ArrayDims<2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("forEachADCoord_3D")
{
    llama::ArrayDims<3> ad_size{3, 3, 3};

    std::vector<llama::ArrayDims<3>> coords;
    llama::forEachADCoord(ad_size, [&](llama::ArrayDims<3> coord) { coords.push_back(coord); });

    CHECK(
        coords
        == std::vector<llama::ArrayDims<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}
