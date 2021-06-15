/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Stopwatch.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto mapping
    = 1; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blobs, does not work yet), 3 tree AoS, 4 tree SoA
constexpr auto problem_size = 64 * 1024 * 1024;
constexpr auto block_size = 256;
constexpr auto steps = 10;

using FP = float;

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
    struct Z{};
} // namespace tag

using Vector = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
// clang-format on

template <std::size_t TProblemSize, std::size_t TElems>
struct AddKernel
{
    template <typename TAcc, typename TView>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const TAcc& acc, TView a, TView b) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        const auto start = ti * TElems;
        const auto end = alpaka::math::min(acc, start + TElems, TProblemSize);

        LLAMA_INDEPENDENT_DATA
        for (auto i = start; i < end; ++i)
        {
            a(i)(tag::X{}) += b(i)(tag::X{});
            a(i)(tag::Y{}) -= b(i)(tag::Y{});
            a(i)(tag::Z{}) *= b(i)(tag::Z{});
        }
    }
};

auto main() -> int
try
{
    // ALPAKA
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;

    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;

    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;
    const DevAcc dev_acc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost dev_host(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(dev_acc);

    // LLAMA
    const auto array_dims = llama::ArrayDims{problem_size};

    const auto mapping = [array_dims]
    {
        if constexpr (mapping == 0)
            return llama::mapping::AoS{array_dims, Vector{}};
        if constexpr (mapping == 1)
            return llama::mapping::SoA{array_dims, Vector{}};
        if constexpr (mapping == 2)
            return llama::mapping::SoA<decltype(array_dims), Vector, true>{array_dims};
        if constexpr (mapping == 3)
            return llama::mapping::tree::Mapping{array_dims, llama::Tuple{}, Vector{}};
        if constexpr (mapping == 4)
            return llama::mapping::tree::Mapping{
                array_dims,
                llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                Vector{}};
    }();

    std::cout << problem_size / 1000 / 1000 << " million vectors\n"
              << problem_size * llama::sizeOf<Vector> * 2 / 1000 / 1000 << " MB on device\n";

    Stopwatch chrono;

    const auto buffer_size = Size(mapping.blobSize(0));

    // allocate buffers
    auto host_buffer_a = alpaka::allocBuf<std::byte, Size>(dev_host, buffer_size);
    auto host_buffer_b = alpaka::allocBuf<std::byte, Size>(dev_host, buffer_size);
    auto dev_buffer_a = alpaka::allocBuf<std::byte, Size>(dev_acc, buffer_size);
    auto dev_buffer_b = alpaka::allocBuf<std::byte, Size>(dev_acc, buffer_size);

    chrono.printAndReset("Alloc");

    // create LLAMA views
    auto host_a = llama::View{mapping, llama::Array{alpaka::getPtrNative(host_buffer_a)}};
    auto host_b = llama::View{mapping, llama::Array{alpaka::getPtrNative(host_buffer_b)}};
    auto dev_a = llama::View{mapping, llama::Array{alpaka::getPtrNative(dev_buffer_a)}};
    auto dev_b = llama::View{mapping, llama::Array{alpaka::getPtrNative(dev_buffer_b)}};

    chrono.printAndReset("Views");

    std::default_random_engine generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    auto seed = distribution(generator);
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < problem_size; ++i)
    {
        host_a(i) = seed + i;
        host_b(i) = seed - i;
    }
    chrono.printAndReset("Init");

    alpaka::memcpy(queue, dev_buffer_a, host_buffer_a, buffer_size);
    alpaka::memcpy(queue, dev_buffer_b, host_buffer_b, buffer_size);

    chrono.printAndReset("Copy H->D");

    constexpr std::size_t hardware_threads = 2; // relevant for OpenMP2Threads
    using Distribution = common::ThreadsElemsDistribution<Acc, block_size, hardware_threads>;
    constexpr std::size_t elem_count = Distribution::elemCount;
    constexpr std::size_t thread_count = Distribution::threadCount;
    const alpaka::Vec<Dim, Size> elems(static_cast<Size>(elem_count));
    const alpaka::Vec<Dim, Size> threads(static_cast<Size>(thread_count));
    constexpr auto inner_count = elem_count * thread_count;
    const alpaka::Vec<Dim, Size> blocks(static_cast<Size>((problem_size + inner_count - 1) / inner_count));

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{blocks, threads, elems};

    for (std::size_t s = 0; s < steps; ++s)
    {
        alpaka::exec<Acc>(queue, workdiv, AddKernel<problem_size, elem_count>{}, dev_a, dev_b);
        chrono.printAndReset("Add kernel");
    }

    alpaka::memcpy(queue, host_buffer_a, dev_buffer_a, buffer_size);
    alpaka::memcpy(queue, host_buffer_b, dev_buffer_b, buffer_size);

    chrono.printAndReset("Copy D->H");

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
