#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

using RecordDim = Vec3D;
using Mapping = llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>;
using Vector = llama::Vector<Mapping>;

TEST_CASE("Vector.ctor.default")
{
    const Vector v;
    CHECK(v.empty());
    CHECK(v.size() == 0);
}

TEST_CASE("Vector.ctor.count")
{
    const Vector v(10);
    CHECK(!v.empty());
    CHECK(v.size() == 10);
}

TEST_CASE("Vector.ctor.count_and_value")
{
    llama::One<RecordDim> p{42};
    const Vector v(10, p);
    CHECK(v.size() == 10);
    for (auto i = 0; i < 10; i++)
        CHECK(v[i] == p);
}

TEST_CASE("Vector.ctor.iterator_pair")
{
    auto view = llama::allocView(Mapping{llama::ArrayDims{10}});
    for (auto i = 0; i < 10; i++)
        view[i] = i;

    const Vector v{view.begin(), view.end()};
    CHECK(v.size() == 10);
    for (auto i = 0; i < 10; i++)
        CHECK(v[i] == i);
}

TEST_CASE("vector.copy_ctor")
{
    llama::One<RecordDim> p{42};
    const Vector v(10, p);

    const Vector v2(v);
    for (auto i = 0; i < 10; i++)
        CHECK(v2[i] == p);
}

TEST_CASE("vector.move_ctor")
{
    llama::One<RecordDim> p{42};
    Vector v(10, p);

    Vector v2(std::move(v));
    for (auto i = 0; i < 10; i++)
        CHECK(v2[i] == p);
    CHECK(v.empty());
}

TEST_CASE("vector.copy_assign")
{
    llama::One<RecordDim> p{42};
    const Vector v(10, p);

    Vector v2;
    v2 = v;
    for (auto i = 0; i < 10; i++)
        CHECK(v2[i] == p);
}

TEST_CASE("vector.move_assign")
{
    llama::One<RecordDim> p{42};
    Vector v(10, p);

    Vector v2;
    v2 = std::move(v);
    for (auto i = 0; i < 10; i++)
        CHECK(v2[i] == p);
    CHECK(v.empty());
}

namespace
{
    auto iota_vec(std::size_t count)
    {
        Vector v(count);
        for (auto i = 0; i < count; i++)
            v[i] = i;
        return v;
    }
} // namespace

TEST_CASE("vector.swap")
{
    llama::One<RecordDim> p{21};
    Vector v1(5, p);

    auto v2 = iota_vec(10);

    swap(v1, v2);

    for (auto i = 0; i < 10; i++)
        CHECK(v1[i] == i);
    for (auto i = 0; i < 5; i++)
        CHECK(v2[i] == p);
}

TEST_CASE("vector.subscript")
{
    auto v = iota_vec(10);
    for (auto i = 0; i < 10; i++)
        CHECK(v[i] == i);
    for (auto i = 0; i < 10; i++)
        CHECK(v.at(i) == i);
    CHECK_THROWS(v.at(10));
}

TEST_CASE("vector.front")
{
    auto v = iota_vec(10);
    CHECK(v.front() == 0);
}

TEST_CASE("vector.back")
{
    auto v = iota_vec(10);
    CHECK(v.back() == 9);
}

TEST_CASE("vector.begin_end")
{
    auto v = iota_vec(10);
    auto b = v.begin();
    auto e = v.end();
    CHECK(*b == 0);
    CHECK(e[-1] == 9);
    for (auto i = 0; b != e; ++b, i++)
        CHECK(*b == i);
    b = v.begin();
    for (auto i = 0; b != e; ++b, i++)
        *b = i * 2;
    for (auto i = 0; i < 10; i++)
        CHECK(v[i] == i * 2);
}

TEST_CASE("vector.cbegin_cend")
{
    auto v = iota_vec(10);
    auto b = v.cbegin();
    auto e = v.cend();
    CHECK(*b == 0);
    CHECK(e[-1] == 9);
    for (auto i = 0; b != e; ++b, i++)
        CHECK(*b == i);
}

TEST_CASE("vector.reserve")
{
    Vector v;
    CHECK(v.capacity() == 0);
    CHECK(v.size() == 0);
    v.reserve(100);
    CHECK(v.capacity() == 100);
    CHECK(v.size() == 0);
}

TEST_CASE("vector.clear_shrink_to_fit")
{
    auto v = iota_vec(10);
    CHECK(v.capacity() == 10);
    CHECK(v.size() == 10);
    v.clear();
    CHECK(v.capacity() == 10);
    CHECK(v.size() == 0);
    v.shrink_to_fit();
    CHECK(v.capacity() == 0);
    CHECK(v.size() == 0);
}

TEST_CASE("vector.insert")
{
    auto v = iota_vec(2);
    CHECK(*(v.insert(v.begin(), llama::One<RecordDim>{42})) == 42);
    CHECK(v.size() == 3);
    CHECK(v[0] == 42);
    CHECK(v[1] == 0);
    CHECK(v[2] == 1);
    CHECK(*(v.insert(v.begin() + 2, llama::One<RecordDim>{43})) == 43);
    CHECK(v.size() == 4);
    CHECK(v[0] == 42);
    CHECK(v[1] == 0);
    CHECK(v[2] == 43);
    CHECK(v[3] == 1);
    CHECK(*(v.insert(v.end(), llama::One<RecordDim>{44})) == 44);
    CHECK(v.size() == 5);
    CHECK(v[0] == 42);
    CHECK(v[1] == 0);
    CHECK(v[2] == 43);
    CHECK(v[3] == 1);
    CHECK(v[4] == 44);
}

TEST_CASE("vector.push_back")
{
    Vector v;
    v.push_back(llama::One<RecordDim>{42});
    CHECK(v.size() == 1);
    CHECK(v[0] == 42);
    v.push_back(llama::One<RecordDim>{43});
    CHECK(v.size() == 2);
    CHECK(v[0] == 42);
    CHECK(v[1] == 43);
}

TEST_CASE("vector.pop_back")
{
    auto v = iota_vec(2);
    v.pop_back();
    CHECK(v.size() == 1);
    CHECK(v[0] == 0);
    v.pop_back();
    CHECK(v.size() == 0);
}

TEST_CASE("vector.resize")
{
    auto v = iota_vec(5);
    v.resize(2);
    CHECK(v.size() == 2);
    CHECK(v[0] == 0);
    CHECK(v[1] == 1);
    v.resize(4);
    CHECK(v.size() == 4);
    CHECK(v[0] == 0);
    CHECK(v[1] == 1);
    CHECK(v[2] == 0);
    CHECK(v[3] == 0);
    v.resize(6, llama::One<RecordDim>{42});
    CHECK(v.size() == 6);
    CHECK(v[0] == 0);
    CHECK(v[1] == 1);
    CHECK(v[2] == 0);
    CHECK(v[3] == 0);
    CHECK(v[4] == 42);
    CHECK(v[5] == 42);
    v.resize(2, llama::One<RecordDim>{42});
    CHECK(v.size() == 2);
    CHECK(v[0] == 0);
    CHECK(v[1] == 1);
    v.resize(0);
    CHECK(v.size() == 0);
}

TEST_CASE("vector.erase")
{
    auto v = iota_vec(4);
    CHECK(*v.erase(v.begin()) == 1);
    CHECK(v.size() == 3);
    CHECK(v[0] == 1);
    CHECK(v[1] == 2);
    CHECK(v[2] == 3);
    CHECK(*v.erase(v.begin() + 1) == 3);
    CHECK(v.size() == 2);
    CHECK(v[0] == 1);
    CHECK(v[1] == 3);
    auto e = v.erase(v.end() - 1);
    CHECK(e == v.end()); // sequence v.end() after v.erase()
    CHECK(v.size() == 1);
    CHECK(v[0] == 1);
}
