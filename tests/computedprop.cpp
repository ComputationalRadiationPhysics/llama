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

    template <typename TArrayDims, typename TRecordDim>
    struct AoSWithComputedNormal : llama::mapping::PackedAoS<TArrayDims, TRecordDim>
    {
        using Base = llama::mapping::PackedAoS<TArrayDims, TRecordDim>;

        template <std::size_t... TRecordCoords>
        static constexpr auto is_computed(llama::RecordCoord<TRecordCoords...>)
        {
            return llama::RecordCoordCommonPrefixIsSame<llama::RecordCoord<TRecordCoords...>, llama::RecordCoord<3>>;
        }

        template <std::size_t... TRecordCoords, typename TBlob>
        constexpr auto compute(
            TArrayDims coord,
            llama::RecordCoord<TRecordCoords...>,
            llama::Array<TBlob, Base::blobCount>& storage_blobs) const
        {
            auto fetch = [storage_blobs](llama::NrAndOffset nr_and_offset) -> double
            { return *reinterpret_cast<double*>(&storage_blobs[nr_and_offset.nr][nr_and_offset.offset]); };

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

            using DC = llama::RecordCoord<TRecordCoords...>;
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
    auto array_dims = llama::ArrayDims<1>{10};
    auto mapping = AoSWithComputedNormal<decltype(array_dims), Triangle>{array_dims};

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
    template <typename TTArrayDims, typename TTRecordDim>
    struct ComputedMapping
    {
        using ArrayDims = TTArrayDims;
        using RecordDim = TTRecordDim;
        static constexpr std::size_t blobCount = 0;

        constexpr ComputedMapping() = default;

        constexpr explicit ComputedMapping(ArrayDims, RecordDim = {})
        {
        }

        template <std::size_t... TRecordCoords>
        static constexpr auto is_computed(llama::RecordCoord<TRecordCoords...>)
        {
            return true;
        }

        template <std::size_t... TRecordCoords, typename TBlob>
        constexpr auto compute(ArrayDims coord, llama::RecordCoord<TRecordCoords...>, llama::Array<TBlob, blobCount>&)
            const -> std::size_t
        {
            return std::reduce(std::begin(coord), std::end(coord), std::size_t{1}, std::multiplies<>{});
        }
    };
} // namespace

TEST_CASE("fully_computed_mapping")
{
    auto array_dims = llama::ArrayDims<3>{10, 10, 10};
    auto mapping = ComputedMapping<decltype(array_dims), Triangle>{array_dims};

    auto view = llama::allocView(mapping);

    using namespace tag;
    CHECK(view(0u, 1u, 2u)(A{}, X{}) == 0);
    CHECK(view(2u, 1u, 2u)(A{}, Y{}) == 4);
    CHECK(view(2u, 5u, 2u)(A{}, Z{}) == 20);
}

namespace
{
    template <
        typename TTArrayDims,
        typename TTRecordDim,
        typename TLinearizeArrayDimsFunctor = llama::mapping::LinearizeArrayDimsCpp>
    struct CompressedBoolMapping
    {
        using ArrayDims = TTArrayDims;
        using RecordDim = TTRecordDim;
        static constexpr std::size_t blob_count = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        constexpr CompressedBoolMapping() = default;

        constexpr explicit CompressedBoolMapping(ArrayDims size) : array_dims_size(size)
        {
        }

        using Word = std::uint64_t;

        struct BoolRef
        {
            Word& word;
            unsigned char bit;

            explicit operator bool() const
            {
                return (word & (Word{1} << bit)) != 0u;
            }

            auto operator=(bool b) -> BoolRef
            {
                word ^= (-Word{static_cast<(anonymous namespace)::CompressedBoolMapping::Word>(b)} ^ word)
                    & (Word{1} << bit);
                return *this;
            }
        };

        constexpr auto blob_size(std::size_t) const -> std::size_t
        {
            llama::forEachLeaf<RecordDim>([](auto coord) constexpr {
                static_assert(std::is_same_v<llama::GetType<RecordDim, decltype(coord)>, bool>);
            });
            constexpr std::size_t word_bytes = sizeof(Word);
            return (TLinearizeArrayDimsFunctor{}.size(array_dims_size) + word_bytes - 1) / word_bytes;
        }

        template <std::size_t... TRecordCoords>
        static constexpr auto is_computed(llama::RecordCoord<TRecordCoords...>)
        {
            return true;
        }

        template <std::size_t... TRecordCoords, typename TBlob>
        constexpr auto compute(
            ArrayDims coord,
            llama::RecordCoord<TRecordCoords...>,
            llama::Array<TBlob, blob_count>& blobs) const -> BoolRef
        {
            const auto bit_offset = TLinearizeArrayDimsFunctor{}(coord, array_dims_size);
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<TRecordCoords...>>;

            constexpr std::size_t word_bits = sizeof(Word) * CHAR_BIT;
            return BoolRef{
                reinterpret_cast<Word&>(blobs[blob][bit_offset / word_bits]),
                static_cast<unsigned char>(bit_offset % word_bits)};
        }

        ArrayDims array_dims_size;
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
    auto array_dims = llama::ArrayDims{8, 8};
    auto mapping = CompressedBoolMapping<decltype(array_dims), BoolRecord>{array_dims};
    STATIC_REQUIRE(decltype(mapping)::blob_count == 3);
    CHECK(mapping.blob_size(0) == 8);
    CHECK(mapping.blob_size(1) == 8);
    CHECK(mapping.blob_size(2) == 8);

    auto view = llama::allocView(mapping);
    for (auto y = 0u; y < 8; y++)
    {
        for (auto x = 0u; x < 8; x++)
        {
            view(y, x)(tag::A{}) = static_cast<bool>(x * y & 1);
            view(y, x)(tag::B{}, tag::X{}) = static_cast<bool>(x & 1);
            view(y, x)(tag::B{}, tag::Y{}) = static_cast<bool>(y & 1);
        }
    }

    for (auto y = 0u; y < 8; y++)
    {
        for (auto x = 0u; x < 8; x++)
        {
            CHECK(view(y, x)(tag::A{}) == static_cast<bool>(x * y & 1));
            CHECK(view(y, x)(tag::B{}, tag::X{}) == static_cast<bool>(x & 1));
            CHECK(view(y, x)(tag::B{}, tag::Y{}) == static_cast<bool>(y & 1));
        }
    }
}
