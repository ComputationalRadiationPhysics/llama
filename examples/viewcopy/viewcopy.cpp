#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"
#include "../common/ttjet_13tev_june2019.hpp"

#include <boost/functional/hash.hpp>
#include <boost/mp11.hpp>
#include <fmt/format.h>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <llama/llama.hpp>
#include <numeric>
#include <omp.h>
#include <string_view>

constexpr auto repetitions = 5;
constexpr auto array_dims = llama::ArrayDims{512, 512, 16};

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

namespace llamaex
{
    using namespace llama;

    template <std::size_t TDim, typename TFunc>
    void parallel_for_each_ad_coord(ArrayDims<TDim> ad_size, std::size_t num_threads, TFunc&& func)
    {
#pragma omp parallel for num_threads(num_threads)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(ad_size[0]); i++)
        {
            if constexpr (TDim > 1)
                forEachADCoord(internal::popFront(ad_size), std::forward<TFunc>(func), static_cast<std::size_t>(i));
            else
                std::forward<TFunc>(func)(ArrayDims<TDim>{static_cast<std::size_t>(i)});
        }
    }
} // namespace llamaex

template <typename TSrcMapping, typename TSrcBlobType, typename TDstMapping, typename TDstBlobType>
void naive_copy(
    const llama::View<TSrcMapping, TSrcBlobType>& src_view,
    llama::View<TDstMapping, TDstBlobType>& dstView,
    std::size_t num_threads = 1)
{
    static_assert(std::is_same_v<typename TSrcMapping::RecordDim, typename TDstMapping::RecordDim>);

    if (src_view.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    llamaex::parallel_for_each_ad_coord(
        src_view.mapping.arrayDims(),
        num_threads,
        [src_view](auto ad) LLAMA_LAMBDA_INLINE
        {
            llama::forEachLeaf<typename TDstMapping::RecordDim>([src_view](auto coord) LLAMA_LAMBDA_INLINE
                                                                { dstView(ad)(coord) = src_view(ad)(coord); });
        });
}

template <typename TSrcMapping, typename TSrcBlobType, typename TDstMapping, typename TDstBlobType>
void std_copy(
    const llama::View<TSrcMapping, TSrcBlobType>& src_view,
    llama::View<TDstMapping, TDstBlobType>& dst_view,
    std::size_t num_threads = 1)
{
    static_assert(std::is_same_v<typename TSrcMapping::RecordDim, typename TDstMapping::RecordDim>);

    if (src_view.mapping.arrayDims() != dst_view.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    std::copy(src_view.begin(), src_view.end(), dst_view.begin());
}

// adapted from: https://stackoverflow.com/a/30386256/1034717
void* memcpy_avx2(void* dst, const void* src, size_t n) noexcept
{
#define ALIGN(ptr, align) (((ptr) + (align) -1) & ~((align) -1))

    auto* d = static_cast<char*>(dst);
    const auto* s = static_cast<const char*>(src);

    // fall back to memcpy() if dst and src are misaligned
    if ((reinterpret_cast<uintptr_t>(d) & 31) != (reinterpret_cast<uintptr_t>(s) & 31))
        return memcpy(d, s, n);

    // align dst/src address multiple of 32
    if ((reinterpret_cast<uintptr_t>(d) & 31) != 0u)
    {
        uintptr_t header_bytes = 32 - (reinterpret_cast<uintptr_t>(d) & 31);
        assert(header_bytes < 32);

        memcpy(d, s, std::min(header_bytes, n));

        d = reinterpret_cast<char*>(ALIGN(reinterpret_cast<uintptr_t>(d), 32));
        s = reinterpret_cast<char*>(ALIGN(reinterpret_cast<uintptr_t>(s), 32));
        n -= std::min(header_bytes, n);
    }

    constexpr auto unroll_factor = 8;
    constexpr auto bytes_per_iteration = 32 * unroll_factor;
    while (n >= bytes_per_iteration)
    {
#pragma unroll
#pragma GCC unroll unrollFactor
        for (auto i = 0; i < unroll_factor; i++)
            _mm256_stream_si256(
                reinterpret_cast<__m256i*>(d) + i,
                _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(s) + i));
        s += bytes_per_iteration;
        d += bytes_per_iteration;
        n -= bytes_per_iteration;
    }

    if (n > 0)
        memcpy(d, s, n);

    return dst;
#undef ALIGN
}

inline void parallel_memcpy(
    std::byte* dst,
    const std::byte* src,
    std::size_t size,
    decltype(std::memcpy) = std::memcpy,
    std::size_t num_threads = 1)
{
    const auto size_per_thread = size / num_threads;
    const auto size_last_thread = size_per_thread + size % num_threads;

#pragma omp parallel num_threads(num_threads)
    {
        const auto id = static_cast<std::size_t>(omp_get_thread_num());
        const auto size_this_thread = id == num_threads - 1 ? size_last_thread : size_per_thread;
        std::memcpy(dst + id * size_per_thread, src + id * size_per_thread, size_this_thread);
    }
}

template <
    bool TReadOpt,
    typename TArrayDims,
    typename TRecordDim,
    std::size_t TLanesSrc,
    std::size_t TLanesDst,
    bool TMbSrc,
    bool TMbDst,
    typename SrcView,
    typename DstView>
void aosoa_copy_internal(const SrcView& src_view, DstView& dst_view, std::size_t num_threads)
{
    if (src_view.mapping.arrayDims() != dst_view.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    constexpr auto src_is_ao_so_a = TLanesSrc != std::numeric_limits<std::size_t>::max();
    constexpr auto dst_is_ao_so_a = TLanesDst != std::numeric_limits<std::size_t>::max();

    static_assert(!src_is_ao_so_a || decltype(src_view.storageBlobs)::rank == 1);
    static_assert(!dst_is_ao_so_a || decltype(dst_view.storageBlobs)::rank == 1);

    const auto array_dims = dst_view.mapping.arrayDims();
    const auto flat_size
        = std::reduce(std::begin(array_dims), std::end(array_dims), std::size_t{1}, std::multiplies<>{});

    // the same as AoSoA::blobNrAndOffset but takes a flat array index
    auto map_ao_so_a = [](std::size_t flat_array_index, auto coord, std::size_t lanes) LLAMA_LAMBDA_INLINE
    {
        const auto block_index = flat_array_index / lanes;
        const auto lane_index = flat_array_index % lanes;
        const auto offset = (llama::sizeOf<TRecordDim> * lanes) * block_index
            + llama::offsetOf<TRecordDim, decltype(coord)> * lanes
            + sizeof(llama::GetType<TRecordDim, decltype(coord)>) * lane_index;
        return offset;
    };
    // the same as SoA::blobNrAndOffset but takes a flat array index
    auto map_so_a = [flat_size](std::size_t flat_array_index, auto coord, bool mb) LLAMA_LAMBDA_INLINE
    {
        const auto blob = mb * llama::flatRecordCoord<TRecordDim, decltype(coord)>;
        const auto offset = !mb * llama::offsetOf<TRecordDim, decltype(coord)> * flat_size
            + sizeof(llama::GetType<TRecordDim, decltype(coord)>) * flat_array_index;
        return llama::NrAndOffset{blob, offset};
    };

    auto mapSrc = [&src_view, &map_ao_so_a, &map_so_a](std::size_t flat_array_index, auto coord) LLAMA_LAMBDA_INLINE
    {
        if constexpr (src_is_ao_so_a)
            return &src_view.storageBlobs[0][0] + map_ao_so_a(flat_array_index, coord, TLanesSrc);
        else
        {
            const auto [blob, off] = map_so_a(flat_array_index, coord, TMbSrc);
            return &src_view.storageBlobs[blob][off];
        }
    };
    auto mapDst = [&dst_view, &map_ao_so_a, &map_so_a](std::size_t flat_array_index, auto coord) LLAMA_LAMBDA_INLINE
    {
        if constexpr (dst_is_ao_so_a)
            return &dst_view.storageBlobs[0][0] + map_ao_so_a(flat_array_index, coord, TLanesDst);
        else
        {
            const auto [blob, off] = map_so_a(flat_array_index, coord, TMbDst);
            return &dst_view.storageBlobs[blob][off];
        }
    };

    constexpr auto L = std::min(TLanesSrc, TLanesDst);
    static_assert(!src_is_ao_so_a || TLanesSrc % L == 0);
    static_assert(!dst_is_ao_so_a || TLanesDst % L == 0);
    if constexpr (TReadOpt)
    {
        // optimized for linear reading
        const auto elements_per_thread
            = src_is_ao_so_a ? flat_size / TLanesSrc / num_threads * TLanesSrc : flat_size / L / num_threads * L;
#pragma omp parallel num_threads(num_threads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto start = id * elements_per_thread;
            const auto stop = id == num_threads - 1 ? flat_size : (id + 1) * elements_per_thread;

            auto copy_l_block = [&](const std::byte*& thread_src, std::size_t dst_index, auto coord) LLAMA_LAMBDA_INLINE
            {
                constexpr auto bytes = L * sizeof(llama::GetType<TRecordDim, decltype(coord)>);
                std::memcpy(mapDst(dst_index, coord), thread_src, bytes);
                thread_src += bytes;
            };
            if constexpr (src_is_ao_so_a)
            {
                auto* threadSrc = mapSrc(start, llama::RecordCoord<>{});
                for (std::size_t i = start; i < stop; i += TLanesSrc)
                    llama::forEachLeaf<TRecordDim>(
                        [copy_l_block](auto coord) LLAMA_LAMBDA_INLINE
                        {
                            for (std::size_t j = 0; j < TLanesSrc; j += L)
                                copy_l_block(threadSrc, i + j, coord);
                        });
            }
            else
            {
                llama::forEachLeaf<TRecordDim>(
                    [copy_l_block](auto coord) LLAMA_LAMBDA_INLINE
                    {
                        auto* thread_src = mapSrc(start, coord);
                        for (std::size_t i = start; i < stop; i += L)
                            copy_l_block(thread_src, i, coord);
                    });
            }
        }
    }
    else
    {
        // optimized for linear writing
        const auto elements_per_thread
            = dst_is_ao_so_a ? ((flat_size / TLanesDst) / num_threads) * TLanesDst : flat_size / L / num_threads * L;
#pragma omp parallel num_threads(num_threads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto start = id * elements_per_thread;
            const auto stop = id == num_threads - 1 ? flat_size : (id + 1) * elements_per_thread;

            auto copy_l_block = [&](std::byte*& thread_dst, std::size_t src_index, auto coord) LLAMA_LAMBDA_INLINE
            {
                constexpr auto bytes = L * sizeof(llama::GetType<TRecordDim, decltype(coord)>);
                std::memcpy(thread_dst, mapSrc(src_index, coord), bytes);
                thread_dst += bytes;
            };
            if constexpr (dst_is_ao_so_a)
            {
                auto* threadDst = mapDst(start, llama::RecordCoord<>{});
                for (std::size_t i = start; i < stop; i += TLanesDst)
                    llama::forEachLeaf<TRecordDim>(
                        [copy_l_block](auto coord) LLAMA_LAMBDA_INLINE
                        {
                            for (std::size_t j = 0; j < TLanesDst; j += L)
                                copy_l_block(threadDst, i + j, coord);
                        });
            }
            else
            {
                llama::forEachLeaf<TRecordDim>(
                    [copy_l_block](auto coord) LLAMA_LAMBDA_INLINE
                    {
                        auto* thread_dst = mapDst(start, coord);
                        for (std::size_t i = start; i < stop; i += L)
                            copy_l_block(thread_dst, i, coord);
                    });
            }
        }
    }
}

template <
    bool TReadOpt,
    typename TArrayDims,
    typename TRecordDim,
    std::size_t TLanesSrc,
    typename TSrcBlobType,
    std::size_t TLanesDst,
    typename TDstBlobType>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<TArrayDims, TRecordDim, TLanesSrc, llama::mapping::LinearizeArrayDimsCpp>,
        TSrcBlobType>& src_view,
    llama::View<
        llama::mapping::AoSoA<TArrayDims, TRecordDim, TLanesDst, llama::mapping::LinearizeArrayDimsCpp>,
        TDstBlobType>& dst_view,
    std::size_t num_threads = 1)
{
    aosoa_copy_internal<TReadOpt, TArrayDims, TRecordDim, TLanesSrc, TLanesDst, false, false>(
        src_view,
        dst_view,
        num_threads);
}

template <
    bool TReadOpt,
    typename TArrayDims,
    typename TRecordDim,
    std::size_t TLanesSrc,
    typename TSrcBlobType,
    bool TDstSeparateBuffers,
    typename TDstBlobType>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<TArrayDims, TRecordDim, TLanesSrc, llama::mapping::LinearizeArrayDimsCpp>,
        TSrcBlobType>& src_view,
    llama::View<
        llama::mapping::SoA<TArrayDims, TRecordDim, TDstSeparateBuffers, llama::mapping::LinearizeArrayDimsCpp>,
        TDstBlobType>& dst_view,
    std::size_t num_threads = 1)
{
    aosoa_copy_internal<
        TReadOpt,
        TArrayDims,
        TRecordDim,
        TLanesSrc,
        std::numeric_limits<std::size_t>::max(),
        false,
        TDstSeparateBuffers>(src_view, dst_view, num_threads);
}

template <
    bool TReadOpt,
    typename TArrayDims,
    typename TRecordDim,
    bool TSrcSeparateBuffers,
    typename TSrcBlobType,
    std::size_t TLanesDst,
    typename TDstBlobType>
void aosoa_copy(
    const llama::View<
        llama::mapping::SoA<TArrayDims, TRecordDim, TSrcSeparateBuffers, llama::mapping::LinearizeArrayDimsCpp>,
        TSrcBlobType>& src_view,
    llama::View<
        llama::mapping::AoSoA<TArrayDims, TRecordDim, TLanesDst, llama::mapping::LinearizeArrayDimsCpp>,
        TDstBlobType>& dst_view,
    std::size_t num_threads = 1)
{
    aosoa_copy_internal<
        TReadOpt,
        TArrayDims,
        TRecordDim,
        std::numeric_limits<std::size_t>::max(),
        TLanesDst,
        TSrcSeparateBuffers,
        false>(src_view, dst_view, num_threads);
}


template <typename TMapping, typename TBlobType>
auto hash(const llama::View<TMapping, TBlobType>& view)
{
    std::size_t acc = 0;
    for (auto ad : llama::ArrayDimsIndexRange{view.mapping.arrayDims()})
        llama::forEachLeaf<typename TMapping::RecordDim>([&](auto coord)
                                                         { boost::hash_combine(acc, view(ad)(coord)); });
    return acc;
}
template <typename TMapping>
auto prepare_view_and_hash(TMapping mapping)
{
    auto view = llama::allocView(mapping);

    auto value = std::size_t{0};
    for (auto ad : llama::ArrayDimsIndexRange{mapping.arrayDims()})
        llama::forEachLeaf<typename TMapping::RecordDim>([&](auto coord) { view(ad)(coord) = value++; });

    const auto check_sum = hash(view);
    return std::tuple{view, check_sum};
}

template <typename TMapping>
inline constexpr auto is_AoSoA = false;

template <typename AD, typename RD, std::size_t L>
inline constexpr auto is_AoSoA<llama::mapping::AoSoA<AD, RD, L>> = true;

auto main() -> int
try
{
    const auto dataSize = std::reduce(array_dims.begin(), array_dims.end(), std::size_t{1}, std::multiplies{})
        * llama::sizeOf<RecordDim>;
    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY");
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;
    fmt::print(
        R"(Data size: {}MiB
Threads: {}
Affinity: {}
)",
        dataSize / 1024 / 1024,
        numThreads,
        affinity);

    std::ofstream plot_file{"viewcopy.sh"};
    plot_file.exceptions(std::ios::badbit | std::ios::failbit);
    plot_file << fmt::format(
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

    plot_file << "\"\"\t\"memcpy\"\t\"memcpy\\\\\\_avx2\"\t\"memcpy(p)\"\t\"memcpy\\\\\\_avx2(p)\"\t\"naive "
                 "copy\"\t\"std::copy\"\t\"aosoa copy(r)\"\t\"aosoa copy(w)\"\t\"naive copy(p)\"\t\"aosoa "
                 "copy(r,p)\"\t\"aosoa copy(w,p)\"\n";

    std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> src(dataSize);

    auto benchmark_memcpy = [plot_file](std::string_view name, auto memcpy)
    {
        std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> dst(dataSize);
        Stopwatch watch;
        for (auto i = 0; i < repetitions; i++)
            memcpy(dst.data(), src.data(), dataSize);
        const auto seconds = watch.printAndReset(name, '\t') / repetitions;
        const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
        std::cout << gbs << "GiB/s\t\n";
        plot_file << gbs << "\t";
    };

    std::cout << "byte[] -> byte[]\n";
    plot_file << "\"byte[] -> byte[]\"\t";
    benchmark_memcpy("memcpy", std::memcpy);
    benchmark_memcpy("memcpy_avx2", memcpy_avx2);
    benchmark_memcpy(
        "memcpy(p)",
        [&](auto* dst, auto* src, auto size) { parallel_memcpy(dst, src, size, std::memcpy, numThreads); });
    benchmark_memcpy(
        "memcpy_avx2(p)",
        [&](auto* dst, auto* src, auto size) { parallel_memcpy(dst, src, size, memcpy_avx2, numThreads); });
    plot_file << "0\t";
    plot_file << "0\t";
    plot_file << "0\t";
    plot_file << "0\t";
    plot_file << "0\t";
    plot_file << "0\t";
    plot_file << "0\t";
    plot_file << "\n";

    auto benchmark_all_copies
        = [plot_file](std::string_view src_name, std::string_view dst_name, auto src_mapping, auto dstMapping)
    {
        std::cout << src_name << " -> " << dst_name << "\n";
        plot_file << "\"" << src_name << " -> " << dst_name << "\"\t";

        auto [srcView, srcHash] = prepare_view_and_hash(src_mapping);

        auto benchmark_copy = [plot_file, src_view = srcView, src_hash = srcHash](std::string_view name, auto copy)
        {
            auto dst_view = llama::allocView(dstMapping);
            Stopwatch watch;
            for (auto i = 0; i < repetitions; i++)
                copy(src_view, dst_view);
            const auto seconds = watch.printAndReset(name, '\t') / repetitions;
            const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
            const auto dst_hash = hash(dst_view);
            std::cout << gbs << "GiB/s\t" << (src_hash == dst_hash ? "" : "\thash BAD ") << "\n";
            plot_file << gbs << "\t";
        };

        plot_file << "0\t";
        plot_file << "0\t";
        plot_file << "0\t";
        plot_file << "0\t";
        benchmark_copy("naive copy", [](const auto& src_view, auto& dst_view) { naive_copy(src_view, dst_view); });
        benchmark_copy("std::copy", [](const auto& src_view, auto& dst_view) { std_copy(src_view, dst_view); });
        constexpr auto one_is_ao_so_a = is_AoSoA<decltype(src_mapping)> || is_AoSoA<decltype(dstMapping)>;
        if constexpr (one_is_ao_so_a)
        {
            benchmark_copy(
                "aosoa copy(r)",
                [](const auto& src_view, auto& dst_view) { aosoa_copy<true>(src_view, dst_view); });
            benchmark_copy(
                "aosoa copy(w)",
                [](const auto& src_view, auto& dst_view) { aosoa_copy<false>(src_view, dst_view); });
        }
        else
        {
            plot_file << "0\t";
            plot_file << "0\t";
        }
        benchmark_copy(
            "naive copy(p)",
            [&](const auto& src_view, auto& dst_view) { naive_copy(src_view, dst_view, numThreads); });
        if constexpr (one_is_ao_so_a)
        {
            benchmark_copy(
                "aosoa_copy(r,p)",
                [&](const auto& src_view, auto& dst_view) { aosoa_copy<true>(src_view, dst_view, numThreads); });
            benchmark_copy(
                "aosoa_copy(w,p)",
                [&](const auto& src_view, auto& dst_view) { aosoa_copy<false>(src_view, dst_view, numThreads); });
        }
        else
        {
            plot_file << "0\t";
            plot_file << "0\t";
        }
        plot_file << "\n";
    };

    const auto packed_ao_s_mapping = llama::mapping::PackedAoS<decltype(array_dims), RecordDim>{array_dims};
    const auto aligned_ao_s_mapping = llama::mapping::AlignedAoS<decltype(array_dims), RecordDim>{array_dims};
    const auto multi_blob_so_a_mapping = llama::mapping::MultiBlobSoA<decltype(array_dims), RecordDim>{array_dims};
    const auto aosoa8_mapping = llama::mapping::AoSoA<decltype(array_dims), RecordDim, 8>{array_dims};
    const auto aosoa32_mapping = llama::mapping::AoSoA<decltype(array_dims), RecordDim, 32>{array_dims};
    const auto aosoa64_mapping = llama::mapping::AoSoA<decltype(array_dims), RecordDim, 64>{array_dims};

    benchmark_all_copies("P AoS", "A AoS", packed_ao_s_mapping, aligned_ao_s_mapping);
    benchmark_all_copies("A AoS", "P AoS", aligned_ao_s_mapping, packed_ao_s_mapping);

    benchmark_all_copies("A AoS", "SoA MB", aligned_ao_s_mapping, multi_blob_so_a_mapping);
    benchmark_all_copies("SoA MB", "A AoS", multi_blob_so_a_mapping, aligned_ao_s_mapping);

    benchmark_all_copies("SoA MB", "AoSoA32", multi_blob_so_a_mapping, aosoa32_mapping);
    benchmark_all_copies("AoSoA32", "SoA MB", aosoa32_mapping, multi_blob_so_a_mapping);

    benchmark_all_copies("AoSoA8", "AoSoA32", aosoa8_mapping, aosoa32_mapping);
    benchmark_all_copies("AoSoA32", "AoSoA8", aosoa32_mapping, aosoa8_mapping);

    benchmark_all_copies("AoSoA8", "AoSoA64", aosoa8_mapping, aosoa64_mapping);
    benchmark_all_copies("AoSoA64", "AoSoA8", aosoa64_mapping, aosoa8_mapping);

    plot_file << R"(EOD
plot $data using 2:xtic(1) ti col, "" using 3 ti col, "" using 4 ti col, "" using 5 ti col, "" using 6 ti col, "" using 7 ti col, "" using 8 ti col, "" using 9 ti col, "" using 10 ti col, "" using 11 ti col, "" using 12 ti col
)";
    std::cout << "Plot with: ./viewcopy.sh\n";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
