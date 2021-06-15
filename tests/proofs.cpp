#include <catch2/catch.hpp>
#include <llama/Proofs.hpp>
#include <llama/llama.hpp>
#include <numeric>

// clang-format off
namespace tag {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Flags {};
} // namespace tag

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, double>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, double>
    >>,
    llama::Field<tag::Weight, float>,
    llama::Field<tag::Momentum, llama::Record<
        llama::Field<tag::X, double>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, double>
    >>,
    llama::Field<tag::Flags, bool[4]>
>;
// clang-format on

TEST_CASE("mapsNonOverlappingly.AoS")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr auto array_dims = ArrayDims{32, 32};
    constexpr auto mapping = llama::mapping::AoS<ArrayDims, Particle>{array_dims};

#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    template <typename TTArrayDims, typename TTRecordDim>
    struct MapEverythingToZero
    {
        using ArrayDims = TTArrayDims;
        using RecordDim = TTRecordDim;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit MapEverythingToZero(ArrayDims size, RecordDim = {}) : m_array_dims_size(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return m_array_dims_size;
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return std::reduce(
                       std::begin(m_array_dims_size),
                       std::end(m_array_dims_size),
                       std::size_t{1},
                       std::multiplies{})
                * llama::sizeOf<RecordDim>;
        }

        template <std::size_t... TDdCs>
        constexpr auto blobNrAndOffset(ArrayDims) const -> llama::NrAndOffset
        {
            return {0, 0};
        }

    private:
        ArrayDims m_array_dims_size;
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.MapEverythingToZero")
{
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDims<1>{1}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDims<1>{2}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDims<2>{32, 32}, Particle{}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    // maps each element of the record dimension into a separate blobs. Each blob stores Modulus elements. If the array
    // dimensions are larger than Modulus, elements are overwritten.
    template <typename TTArrayDims, typename TTRecordDim, std::size_t TModulus>
    struct ModulusMapping
    {
        using ArrayDims = TTArrayDims;
        using RecordDim = TTRecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit ModulusMapping(ArrayDims size, RecordDim = {}) : m_array_dims_size(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return m_array_dims_size;
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return TModulus * llama::sizeOf<RecordDim>;
        }

        template <std::size_t... TDdCs>
        constexpr auto blobNrAndOffset(ArrayDims coord) const -> llama::NrAndOffset
        {
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<TDdCs...>>;
            const auto offset = (llama::mapping::LinearizeArrayDimsCpp{}(coord, m_array_dims_size) % TModulus)
                * sizeof(llama::GetType<RecordDim, llama::RecordCoord<TDdCs...>>);
            return {blob, offset};
        }

    private:
        ArrayDims m_array_dims_size;
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.ModulusMapping")
{
    using Modulus10Mapping = ModulusMapping<llama::ArrayDims<1>, Particle, 10>;

#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{1}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{9}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{10}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{11}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{25}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

TEST_CASE("maps.ModulusMapping")
{
    constexpr auto array_dims = llama::ArrayDims{128};
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::AoS{array_dims, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<8>(llama::mapping::AoS{array_dims, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<16>(llama::mapping::AoS{array_dims, Particle{}}));

    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::SoA{array_dims, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<8>(llama::mapping::SoA{array_dims, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<16>(llama::mapping::SoA{array_dims, Particle{}}));

    STATIC_REQUIRE(
        llama::mapsPiecewiseContiguous<1>(llama::mapping::AoSoA<decltype(array_dims), Particle, 8>{array_dims}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<8>(
        llama::mapping::AoSoA<decltype(array_dims), Particle, 8>{array_dims, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<16>(
        llama::mapping::AoSoA<decltype(array_dims), Particle, 8>{array_dims, Particle{}}));
}
