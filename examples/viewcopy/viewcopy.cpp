#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"
#include "../common/ttjet_13tev_june2019.hpp"

#include <boost/functional/hash.hpp>
#include <fmt/format.h>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <llama/llama.hpp>
#include <numeric>
#include <omp.h>
#include <string_view>

constexpr auto REPETITIONS = 5;
constexpr auto arrayDims = llama::ArrayDims{512, 512, 16};

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
} // namespace tag

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, float>,
        llama::Field<tag::Z, float>
    >>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, float>,
        llama::Field<tag::Z, float>
    >>,
    llama::Field<tag::Mass, float>
>;
// clang-format on

// using RecordDim = Particle;
using RecordDim = boost::mp11::mp_take_c<Event, 20>;
// using RecordDim = Event; // WARN: expect long compilation time

template <typename SrcMapping, typename SrcBlobType, typename DstMapping, typename DstBlobType>
void std_copy(const llama::View<SrcMapping, SrcBlobType>& srcView, llama::View<DstMapping, DstBlobType>& dstView)
{
    static_assert(std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>);

    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    std::copy(srcView.begin(), srcView.end(), dstView.begin());
}

#ifdef __AVX2__
// adapted from: https://stackoverflow.com/a/30386256/1034717
auto memcpy_avx2(void* dst, const void* src, size_t n) noexcept -> void*
{
    auto* d = static_cast<std::byte*>(dst);
    const auto* s = static_cast<const std::byte*>(src);

    // fall back to memcpy() if dst and src are misaligned
    const auto lowerDstBits = reinterpret_cast<uintptr_t>(d) & 31u;
    if (lowerDstBits != (reinterpret_cast<uintptr_t>(s) & 31u))
        return memcpy(d, s, n);

    // align dst/src address multiple of 32
    if (lowerDstBits != 0u)
    {
        const auto header_bytes = std::min(static_cast<size_t>(32 - lowerDstBits), n);
        memcpy(d, s, header_bytes);
        d += header_bytes;
        s += header_bytes;
        n -= header_bytes;
    }

    constexpr auto unrollFactor = 8;
    constexpr auto bytesPerIteration = 32 * unrollFactor;
    while (n >= bytesPerIteration)
    {
#    pragma unroll
#    pragma GCC unroll unrollFactor
        for (auto i = 0; i < unrollFactor; i++)
            _mm256_stream_si256(
                reinterpret_cast<__m256i*>(d) + i,
                _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(s) + i));
        s += bytesPerIteration;
        d += bytesPerIteration;
        n -= bytesPerIteration;
    }

    if (n > 0)
        memcpy(d, s, n);

    return dst;
}
#endif

template <typename Mapping, typename BlobType>
auto hash(const llama::View<Mapping, BlobType>& view)
{
    std::size_t acc = 0;
    for (auto ad : llama::ArrayDimsIndexRange{view.mapping.arrayDims()})
        llama::forEachLeaf<typename Mapping::RecordDim>([&](auto coord) { boost::hash_combine(acc, view(ad)(coord)); });
    return acc;
}
template <typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocView(mapping);

    auto value = std::size_t{0};
    for (auto ad : llama::ArrayDimsIndexRange{mapping.arrayDims()})
        llama::forEachLeaf<typename Mapping::RecordDim>([&](auto coord) { view(ad)(coord) = value++; });

    const auto checkSum = hash(view);
    return std::tuple{view, checkSum};
}

template <typename Mapping>
inline constexpr auto is_AoSoA = false;

template <typename AD, typename RD, std::size_t L>
inline constexpr auto is_AoSoA<llama::mapping::AoSoA<AD, RD, L>> = true;

auto main() -> int
try
{
    const auto dataSize
        = std::reduce(arrayDims.begin(), arrayDims.end(), std::size_t{1}, std::multiplies{}) * llama::sizeOf<RecordDim>;
    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY"); // NOLINT(concurrency-mt-unsafe)
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;
    fmt::print(
        R"(Data size: {}MiB
Threads: {}
Affinity: {}
)",
        dataSize / 1024 / 1024,
        numThreads,
        affinity);

    std::ofstream plotFile{"viewcopy.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# threads: {} affinity: {}
set title "viewcopy CPU {}MiB particles on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 4
set ylabel "throughput [GiB/s]"
$data << EOD
)",
        numThreads,
        affinity,
        dataSize / 1024 / 1024,
        common::hostname());

    plotFile << "\"\"\t\"memcpy\"\t\"memcpy\\\\\\_avx2\"\t\"memcpy(p)\"\t\"memcpy\\\\\\_avx2(p)\"\t\"naive "
                "copy\"\t\"std::copy\"\t\"aosoa copy(r)\"\t\"aosoa copy(w)\"\t\"naive copy(p)\"\t\"aosoa "
                "copy(r,p)\"\t\"aosoa copy(w,p)\"\n";

    std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> src(dataSize);

    auto benchmarkMemcpy = [&](std::string_view name, auto memcpy)
    {
        std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> dst(dataSize);
        Stopwatch watch;
        for (auto i = 0; i < REPETITIONS; i++)
            memcpy(dst.data(), src.data(), dataSize);
        const auto seconds = watch.printAndReset(name, '\t') / REPETITIONS;
        const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
        std::cout << gbs << "GiB/s\t\n";
        plotFile << gbs << "\t";
    };

    std::cout << "byte[] -> byte[]\n";
    plotFile << "\"byte[] -> byte[]\"\t";
    benchmarkMemcpy("memcpy", std::memcpy);
#ifdef __AVX2__
    benchmarkMemcpy("memcpy_avx2", memcpy_avx2);
#else
    plotFile << "0\t";
#endif
    benchmarkMemcpy(
        "memcpy(p)",
        [&](auto* dst, auto* src, auto size)
        {
#pragma omp parallel
            llama::internal::parallel_memcpy(dst, src, size, omp_get_thread_num(), omp_get_num_threads(), std::memcpy);
        });
#ifdef __AVX2__
    benchmarkMemcpy(
        "memcpy_avx2(p)",
        [&](auto* dst, auto* src, auto size)
        {
#    pragma omp parallel
            llama::internal::parallel_memcpy(dst, src, size, omp_get_thread_num(), omp_get_num_threads(), memcpy_avx2);
        });
#else
    plotFile << "0\t";
#endif
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "\n";

    auto benchmarkAllCopies = [&](std::string_view srcName, std::string_view dstName, auto srcMapping, auto dstMapping)
    {
        std::cout << srcName << " -> " << dstName << "\n";
        plotFile << "\"" << srcName << " -> " << dstName << "\"\t";

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);

        auto benchmarkCopy = [&, srcView = srcView, srcHash = srcHash](std::string_view name, auto copy)
        {
            auto dstView = llama::allocView(dstMapping);
            Stopwatch watch;
            for (auto i = 0; i < REPETITIONS; i++)
                copy(srcView, dstView);
            const auto seconds = watch.printAndReset(name, '\t') / REPETITIONS;
            const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
            const auto dstHash = hash(dstView);
            std::cout << gbs << "GiB/s\t" << (srcHash == dstHash ? "" : "\thash BAD ") << "\n";
            plotFile << gbs << "\t";
        };

        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        benchmarkCopy("naive copy", [](const auto& srcView, auto& dstView) { llama::fieldWiseCopy(srcView, dstView); });
        benchmarkCopy("std::copy", [](const auto& srcView, auto& dstView) { std_copy(srcView, dstView); });
        constexpr auto oneIsAoSoA = is_AoSoA<decltype(srcMapping)> || is_AoSoA<decltype(dstMapping)>;
        if constexpr (oneIsAoSoA)
        {
            benchmarkCopy(
                "aosoa copy(r)",
                [](const auto& srcView, auto& dstView) { llama::aosoaCommonBlockCopy(srcView, dstView, true); });
            benchmarkCopy(
                "aosoa copy(w)",
                [](const auto& srcView, auto& dstView) { llama::aosoaCommonBlockCopy(srcView, dstView, false); });
        }
        else
        {
            plotFile << "0\t";
            plotFile << "0\t";
        }
        benchmarkCopy(
            "naive copy(p)",
            [&](const auto& srcView, auto& dstView)
            {
#pragma omp parallel
                // NOLINTNEXTLINE(openmp-exception-escape)
                llama::fieldWiseCopy(srcView, dstView, omp_get_thread_num(), omp_get_num_threads());
            });
        if constexpr (oneIsAoSoA)
        {
            benchmarkCopy(
                "aosoa_copy(r,p)",
                [&](const auto& srcView, auto& dstView)
                {
#pragma omp parallel
                    // NOLINTNEXTLINE(openmp-exception-escape)
                    llama::aosoaCommonBlockCopy(srcView, dstView, true, omp_get_thread_num(), omp_get_num_threads());
                });
            benchmarkCopy(
                "aosoa_copy(w,p)",
                [&](const auto& srcView, auto& dstView)
                {
#pragma omp parallel
                    // NOLINTNEXTLINE(openmp-exception-escape)
                    llama::aosoaCommonBlockCopy(srcView, dstView, false, omp_get_thread_num(), omp_get_num_threads());
                });
        }
        else
        {
            plotFile << "0\t";
            plotFile << "0\t";
        }
        plotFile << "\n";
    };

    const auto packedAoSMapping = llama::mapping::PackedAoS<decltype(arrayDims), RecordDim>{arrayDims};
    const auto alignedAoSMapping = llama::mapping::AlignedAoS<decltype(arrayDims), RecordDim>{arrayDims};
    const auto multiBlobSoAMapping = llama::mapping::MultiBlobSoA<decltype(arrayDims), RecordDim>{arrayDims};
    const auto aosoa8Mapping = llama::mapping::AoSoA<decltype(arrayDims), RecordDim, 8>{arrayDims};
    const auto aosoa32Mapping = llama::mapping::AoSoA<decltype(arrayDims), RecordDim, 32>{arrayDims};
    const auto aosoa64Mapping = llama::mapping::AoSoA<decltype(arrayDims), RecordDim, 64>{arrayDims};

    benchmarkAllCopies("P AoS", "A AoS", packedAoSMapping, alignedAoSMapping);
    benchmarkAllCopies("A AoS", "P AoS", alignedAoSMapping, packedAoSMapping);

    benchmarkAllCopies("A AoS", "SoA MB", alignedAoSMapping, multiBlobSoAMapping);
    benchmarkAllCopies("SoA MB", "A AoS", multiBlobSoAMapping, alignedAoSMapping);

    benchmarkAllCopies("SoA MB", "AoSoA32", multiBlobSoAMapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA32", "SoA MB", aosoa32Mapping, multiBlobSoAMapping);

    benchmarkAllCopies("AoSoA8", "AoSoA32", aosoa8Mapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA32", "AoSoA8", aosoa32Mapping, aosoa8Mapping);

    benchmarkAllCopies("AoSoA8", "AoSoA64", aosoa8Mapping, aosoa64Mapping);
    benchmarkAllCopies("AoSoA64", "AoSoA8", aosoa64Mapping, aosoa8Mapping);

    plotFile << R"(EOD
plot $data using 2:xtic(1) ti col, "" using 3 ti col, "" using 4 ti col, "" using 5 ti col, "" using 6 ti col, "" using 7 ti col, "" using 8 ti col, "" using 9 ti col, "" using 10 ti col, "" using 11 ti col, "" using 12 ti col
)";
    std::cout << "Plot with: ./viewcopy.sh\n";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
