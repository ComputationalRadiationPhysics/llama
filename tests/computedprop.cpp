#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <numeric>

namespace
{
    using Triangle = llama::Record<
        llama::Field<tag::A, Vec3D>,
        llama::Field<tag::B, Vec3D>,
        llama::Field<tag::C, Vec3D>,
        llama::Field<tag::Normal, Vec3D>>;

    template <typename ArrayDims, typename RecordDim>
    struct AoSWithComputedNormal : llama::mapping::PackedAoS<ArrayDims, RecordDim>
    {
        using Base = llama::mapping::PackedAoS<ArrayDims, RecordDim>;

        template <std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return llama::RecordCoordCommonPrefixIsSame<llama::RecordCoord<RecordCoords...>, llama::RecordCoord<3>>;
        }

        template <std::size_t... RecordCoords, std::size_t N, typename Blob>
        constexpr auto compute(
            ArrayDims coord,
            llama::Array<std::size_t, N> dynamicArrayExtents,
            llama::RecordCoord<RecordCoords...>,
            llama::Array<Blob, Base::blobCount>& storageBlobs) const
        {
            auto fetch = [&](llama::NrAndOffset nrAndOffset) -> double
            { return *reinterpret_cast<double*>(&storageBlobs[nrAndOffset.nr][nrAndOffset.offset]); };

            const auto ax = fetch(Base::template blobNrAndOffset<0, 0>(coord));
            const auto ay = fetch(Base::template blobNrAndOffset<0, 1>(coord));
            const auto az = fetch(Base::template blobNrAndOffset<0, 2>(coord));
            const auto bx = fetch(Base::template blobNrAndOffset<1, 0>(coord));
            const auto by = fetch(Base::template blobNrAndOffset<1, 1>(coord));
            const auto bz = fetch(Base::template blobNrAndOffset<1, 2>(coord));
            const auto cx = fetch(Base::template blobNrAndOffset<2, 0>(coord));
            const auto cy = fetch(Base::template blobNrAndOffset<2, 1>(coord));
            const auto cz = fetch(Base::template blobNrAndOffset<2, 2>(coord));

            const auto e1x = bx - ax;
            const auto e1y = by - ay;
            const auto e1z = bz - az;
            const auto e2x = cx - ax;
            const auto e2y = cy - ay;
            const auto e2z = cz - az;

            const auto crossx = e1y * e2z - e1z * e2y;
            const auto crossy = -(e1x * e2z - e1z * e2x);
            const auto crossz = e1x * e2y - e1y * e2x;

            const auto length = std::sqrt(crossx * crossx + crossy * crossy + crossz * crossz);

            const auto normalx = crossx / length;
            const auto normaly = crossy / length;
            const auto normalz = crossz / length;

            using DC = llama::RecordCoord<RecordCoords...>;
            if constexpr (std::is_same_v<DC, llama::RecordCoord<3, 0>>)
                return normalx;
            if constexpr (std::is_same_v<DC, llama::RecordCoord<3, 1>>)
                return normaly;
            if constexpr (std::is_same_v<DC, llama::RecordCoord<3, 2>>)
                return normalz;
            // if constexpr (std::is_same_v<DC, llama::RecordCoord<3>>)
            //{
            //    llama::One<llama::GetType<RecordDim, llama::RecordCoord<3>>> normal;
            //    normal(llama::RecordCoord<0>{}) = normalx;
            //    normal(llama::RecordCoord<1>{}) = normaly;
            //    normal(llama::RecordCoord<2>{}) = normalz;
            //    return normal;
            //}
        }
    };
} // namespace

TEST_CASE("computedprop")
{
    auto arrayDims = llama::ArrayDims<1>{10};
    auto mapping = AoSWithComputedNormal<decltype(arrayDims), Triangle>{arrayDims};

    STATIC_REQUIRE(mapping.blobCount == 1);
    CHECK(mapping.blobSize(0) == 10 * 12 * sizeof(double));

    auto view = llama::allocView(mapping);

    using namespace tag;
    view(5u)(A{}, X{}) = 0.0f;
    view(5u)(A{}, Y{}) = 0.0f;
    view(5u)(A{}, Z{}) = 0.0f;
    view(5u)(B{}, X{}) = 5.0f;
    view(5u)(B{}, Y{}) = 0.0f;
    view(5u)(B{}, Z{}) = 0.0f;
    view(5u)(C{}, X{}) = 0.0f;
    view(5u)(C{}, Y{}) = 3.0f;
    view(5u)(C{}, Z{}) = 0.0f;
    const auto nx = view(5u)(Normal{}, X{});
    const auto ny = view(5u)(Normal{}, Y{});
    const auto nz = view(5u)(Normal{}, Z{});
    CHECK(nx == Approx(0.0f));
    CHECK(ny == Approx(0.0f));
    CHECK(nz == Approx(1.0f));
}

namespace
{
    // Maps accesses to the product of the ArrayDims coord.
    template <typename T_ArrayDims, typename T_RecordDim>
    struct ComputedMapping
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = 0;

        constexpr ComputedMapping() = default;

        constexpr explicit ComputedMapping(ArrayDims, RecordDim = {})
        {
        }

        template <std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template <std::size_t... RecordCoords, std::size_t N, typename Blob>
        constexpr auto compute(
            ArrayDims coord,
            llama::Array<std::size_t, N> dynamicArrayExtents,
            llama::RecordCoord<RecordCoords...>,
            llama::Array<Blob, blobCount>&) const -> std::size_t
        {
            return std::reduce(std::begin(coord), std::end(coord), std::size_t{1}, std::multiplies<>{});
        }
    };
} // namespace

TEST_CASE("fully_computed_mapping")
{
    auto arrayDims = llama::ArrayDims<3>{10, 10, 10};
    auto mapping = ComputedMapping<decltype(arrayDims), Triangle>{arrayDims};

    auto view = llama::allocView(mapping);

    using namespace tag;
    CHECK(view(0u, 1u, 2u)(A{}, X{}) == 0);
    CHECK(view(2u, 1u, 2u)(A{}, Y{}) == 4);
    CHECK(view(2u, 5u, 2u)(A{}, Z{}) == 20);
}

namespace
{
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        typename LinearizeArrayDimsFunctor = llama::mapping::LinearizeArrayDimsCpp>
    struct CompressedBoolMapping
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        constexpr CompressedBoolMapping() = default;

        constexpr explicit CompressedBoolMapping(ArrayDims size) : arrayDimsSize(size)
        {
        }

        using Word = std::uint64_t;

        struct BoolRef
        {
            Word& word;
            unsigned char bit;

            operator bool() const // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
            {
                return (word & (Word{1} << bit)) != 0;
            }

            auto operator=(bool b) -> BoolRef&
            {
                word ^= (-static_cast<Word>(b) ^ word) & (Word{1} << bit);
                return *this;
            }
        };

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            llama::forEachLeaf<RecordDim>([](auto coord) constexpr {
                static_assert(std::is_same_v<llama::GetType<RecordDim, decltype(coord)>, bool>);
            });
            constexpr std::size_t wordBytes = sizeof(Word);
            return (LinearizeArrayDimsFunctor{}.size(arrayDimsSize) + wordBytes - 1) / wordBytes;
        }

        template <std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template <std::size_t... RecordCoords, std::size_t N, typename Blob>
        constexpr auto compute(
            ArrayDims coord,
            llama::Array<std::size_t, N> dynamicArrayExtents,
            llama::RecordCoord<RecordCoords...>,
            llama::Array<Blob, blobCount>& blobs) const -> BoolRef
        {
            const auto bitOffset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize);
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;

            constexpr std::size_t wordBits = sizeof(Word) * CHAR_BIT;
            return BoolRef{
                reinterpret_cast<Word&>(blobs[blob][bitOffset / wordBits]),
                static_cast<unsigned char>(bitOffset % wordBits)};
        }

        ArrayDims arrayDimsSize;
    };

    // clang-format off
    using BoolRecord = llama::Record<
        llama::Field<tag::A, bool>,
        llama::Field<tag::B, llama::Record<
            llama::Field<tag::X, bool>,
            llama::Field<tag::Y, bool>
        >>
    >;
    // clang-format on
} // namespace

TEST_CASE("compressed_bools")
{
    auto arrayDims = llama::ArrayDims{8, 8};
    auto mapping = CompressedBoolMapping<decltype(arrayDims), BoolRecord>{arrayDims};
    STATIC_REQUIRE(decltype(mapping)::blobCount == 3);
    CHECK(mapping.blobSize(0) == 8);
    CHECK(mapping.blobSize(1) == 8);
    CHECK(mapping.blobSize(2) == 8);

    auto view = llama::allocView(mapping);
    for (auto y = 0u; y < 8; y++)
    {
        for (auto x = 0u; x < 8; x++)
        {
            view(y, x)(tag::A{}) = static_cast<bool>(x * y & 1u);
            view(y, x)(tag::B{}, tag::X{}) = static_cast<bool>(x & 1u);
            view(y, x)(tag::B{}, tag::Y{}) = static_cast<bool>(y & 1u);
        }
    }

    for (auto y = 0u; y < 8; y++)
    {
        for (auto x = 0u; x < 8; x++)
        {
            CHECK(view(y, x)(tag::A{}) == static_cast<bool>(x * y & 1u));
            CHECK(view(y, x)(tag::B{}, tag::X{}) == static_cast<bool>(x & 1u));
            CHECK(view(y, x)(tag::B{}, tag::Y{}) == static_cast<bool>(y & 1u));
        }
    }
}
