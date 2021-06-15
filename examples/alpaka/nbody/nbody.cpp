#include "../../common/Stopwatch.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <string>
#include <utility>

using FP = float;

constexpr auto problem_size = 16 * 1024; ///< total number of particles
constexpr auto steps = 5; ///< number of steps to calculate
constexpr auto timestep = FP{0.0001};
constexpr auto allow_rsqrt = true; // rsqrt can be way faster, but less accurate

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        error Cannot enable CUDA together with other backends, because nvcc cannot parse the Vc header, sorry :/
#    endif
// nvcc fails to compile Vc headers even if nothing is used from there, so we need to conditionally include it
#    include <Vc/Vc>
constexpr auto desired_elements_per_thread = Vc::float_v::size();
constexpr auto threads_per_block = 1;
constexpr auto aosoa_lanes = Vc::float_v::size(); // vectors
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
constexpr auto DESIRED_ELEMENTS_PER_THREAD = 1;
constexpr auto THREADS_PER_BLOCK = 256;
constexpr auto AOSOA_LANES = 32; // coalesced memory access
#else
#    error "Unsupported backend"
#endif

// makes our life easier for now
static_assert(problem_size % (desired_elements_per_thread * threads_per_block) == 0);

constexpr FP ep_s2 = 0.01;

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
}

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

enum Mapping
{
    ao_s,
    so_a,
    ao_so_a
};

namespace stdext
{
    LLAMA_FN_HOST_ACC_INLINE FP rsqrt(FP f)
    {
        return 1.0f / std::sqrt(f);
    }
} // namespace stdext

// FIXME: this makes assumptions that there are always float_v::size() many values blocked in the LLAMA view
template <typename TVec>
LLAMA_FN_HOST_ACC_INLINE auto load(const FP& src)
{
    if constexpr (std::is_same_v<TVec, FP>)
        return src;
    else
        return TVec(&src);
}

template <typename TVec>
LLAMA_FN_HOST_ACC_INLINE auto broadcast(const FP& src)
{
    return TVec(src);
}

template <typename TVec>
LLAMA_FN_HOST_ACC_INLINE auto store(FP& dst, TVec v)
{
    if constexpr (std::is_same_v<TVec, FP>)
        dst = v;
    else
        v.store(&dst);
}

template <std::size_t TElems>
struct VecType
{
    // TODO(bgruber): we need a vector type that also works on GPUs
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Type = Vc::SimdArray<FP, TElems>;
#endif
};
template <>
struct VecType<1>
{
    using Type = FP;
};

template <std::size_t TElems, typename TViewParticleI, typename TVirtualParticleJ>
LLAMA_FN_HOST_ACC_INLINE void p_p_interaction(TViewParticleI pi, TVirtualParticleJ pj)
{
    using Vec = typename VecType<TElems>::type;

    using std::sqrt;
    using stdext::rsqrt;
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Vc::rsqrt;
    using Vc::sqrt;
#endif

    const Vec xdistance = load<Vec>(pi(tag::Pos{}, tag::X{})) - broadcast<Vec>(pj(tag::Pos{}, tag::X{}));
    const Vec ydistance = load<Vec>(pi(tag::Pos{}, tag::Y{})) - broadcast<Vec>(pj(tag::Pos{}, tag::Y{}));
    const Vec zdistance = load<Vec>(pi(tag::Pos{}, tag::Z{})) - broadcast<Vec>(pj(tag::Pos{}, tag::Z{}));
    const Vec xdistance_sqr = xdistance * xdistance;
    const Vec ydistance_sqr = ydistance * ydistance;
    const Vec zdistance_sqr = zdistance * zdistance;
    const Vec dist_sqr = +ep_s2 + xdistance_sqr + ydistance_sqr + zdistance_sqr;
    const Vec dist_sixth = dist_sqr * dist_sqr * dist_sqr;
    const Vec inv_dist_cube = allow_rsqrt ? rsqrt(dist_sixth) : (1.0f / sqrt(dist_sixth));
    const Vec sts = broadcast<Vec>(pj(tag::Mass())) * inv_dist_cube * timestep;
    store<Vec>(pi(tag::Vel{}, tag::X{}), xdistance_sqr * sts + load<Vec>(pi(tag::Vel{}, tag::X{})));
    store<Vec>(pi(tag::Vel{}, tag::Y{}), ydistance_sqr * sts + load<Vec>(pi(tag::Vel{}, tag::Y{})));
    store<Vec>(pi(tag::Vel{}, tag::Z{}), zdistance_sqr * sts + load<Vec>(pi(tag::Vel{}, tag::Z{})));
}

template <std::size_t TProblemSize, std::size_t TElems, std::size_t TBlockSize, Mapping TMappingSm>
struct UpdateKernel
{
    template <typename TAcc, typename TView>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const TAcc& acc, TView particles) const
    {
        auto shared_view = [&]
        {
            // if there is only 1 thread per block, use stack instead of shared memory
            if constexpr (TBlockSize == 1)
                return llama::allocViewStack<TView::ArrayDims::rank, typename TView::RecordDim>();
            else
            {
                constexpr auto shared_mapping = []
                {
                    constexpr auto array_dims = llama::ArrayDims{TBlockSize};
                    if constexpr (TMappingSm == ao_s)
                        return llama::mapping::AoS{array_dims, Particle{}};
                    if constexpr (TMappingSm == so_a)
                        return llama::mapping::SoA{array_dims, Particle{}};
                    if constexpr (TMappingSm == ao_so_a)
                        return llama::mapping::AoSoA<decltype(array_dims), Particle, aosoa_lanes>{array_dims};
                }();
                static_assert(decltype(shared_mapping)::blobCount == 1);

                constexpr auto shared_mem_size = llama::sizeOf<typename TView::RecordDim> * TBlockSize;
                auto& shared_mem = alpaka::declareSharedVar<std::byte[shared_mem_size], __COUNTER__>(acc);
                return llama::View{shared_mapping, llama::Array<std::byte*, 1>{&shared_mem[0]}};
            }
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        // TODO(bgruber): we could optimize here, because only velocity is ever updated
        auto pi = [&]
        {
            constexpr auto array_dims = llama::ArrayDims{TElems};
            constexpr auto mapping
                = llama::mapping::SoA<typename TView::ArrayDims, typename TView::RecordDim, false>{array_dims};
            constexpr auto blob_alloc = llama::bloballoc::Stack<llama::sizeOf<typename TView::RecordDim> * TElems>{};
            return llama::allocView(mapping, blob_alloc);
        }();
        // TODO(bgruber): vector load
        LLAMA_INDEPENDENT_DATA
        for (auto e = 0u; e < TElems; e++)
            pi(e) = particles(ti * TElems + e);

        LLAMA_INDEPENDENT_DATA
        for (std::size_t block_offset = 0; block_offset < TProblemSize; block_offset += TBlockSize)
        {
            LLAMA_INDEPENDENT_DATA
            for (auto j = tbi; j < TBlockSize; j += threads_per_block)
                shared_view(j) = particles(block_offset + j);
            alpaka::syncBlockThreads(acc);

            LLAMA_INDEPENDENT_DATA
            for (auto j = std::size_t{0}; j < TBlockSize; ++j)
                p_p_interaction<TElems>(pi(0u), shared_view(j));
            alpaka::syncBlockThreads(acc);
        }
        // TODO(bgruber): vector store
        LLAMA_INDEPENDENT_DATA
        for (auto e = 0u; e < TElems; e++)
            particles(ti * TElems + e) = pi(e);
    }
};

template <std::size_t TProblemSize, std::size_t TElems>
struct MoveKernel
{
    template <typename TAcc, typename TView>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const TAcc& acc, TView particles) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * TElems;

        using Vec = typename VecType<TElems>::type;
        store<Vec>(
            particles(i)(tag::Pos{}, tag::X{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::X{})) + load<Vec>(particles(i)(tag::Vel{}, tag::X{})) * timestep);
        store<Vec>(
            particles(i)(tag::Pos{}, tag::Y{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::Y{})) + load<Vec>(particles(i)(tag::Vel{}, tag::Y{})) * timestep);
        store<Vec>(
            particles(i)(tag::Pos{}, tag::Z{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::Z{})) + load<Vec>(particles(i)(tag::Vel{}, tag::Z{})) * timestep);
    }
};

template <template <typename, typename> typename TAccTemplate, Mapping TMappingGm, Mapping TMappingSm>
void run(std::ostream& plot_file)
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = AccTemplate<Dim, Size>;
    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto mapping_name = [](int m) -> std::string
    {
        if (m == 0)
            return "AoS";
        if (m == 1)
            return "SoA";
        if (m == 2)
            return "AoSoA" + std::to_string(aosoa_lanes);
        std::abort();
    };
    const auto title = "GM " + mapping_name(TMappingGm) + " SM " + mapping_name(TMappingSm);
    std::cout << '\n' << title << '\n';

    const DevAcc dev_acc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost dev_host(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(dev_acc);

    auto mapping = []
    {
        const auto array_dims = llama::ArrayDims{problem_size};
        if constexpr (TMappingGm == ao_s)
            return llama::mapping::AoS{array_dims, Particle{}};
        if constexpr (TMappingGm == so_a)
            return llama::mapping::SoA{array_dims, Particle{}};
        // if constexpr (MappingGM == 2)
        //    return llama::mapping::SoA<decltype(arrayDims), Particle, true>{arrayDims};
        if constexpr (TMappingGm == ao_so_a)
            return llama::mapping::AoSoA<decltype(array_dims), Particle, aosoa_lanes>{array_dims};
    }();

    Stopwatch watch;

    const auto buffer_size = Size(mapping.blobSize(0));

    auto host_buffer = alpaka::allocBuf<std::byte, Size>(dev_host, buffer_size);
    auto acc_buffer = alpaka::allocBuf<std::byte, Size>(dev_acc, buffer_size);

    watch.printAndReset("alloc");

    auto host_view = llama::View{mapping, llama::Array{alpaka::getPtrNative(host_buffer)}};
    auto acc_view = llama::View{mapping, llama::Array{alpaka::getPtrNative(acc_buffer)}};

    watch.printAndReset("views");

    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for (std::size_t i = 0; i < problem_size; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(generator);
        p(tag::Pos(), tag::Y()) = distribution(generator);
        p(tag::Pos(), tag::Z()) = distribution(generator);
        p(tag::Vel(), tag::X()) = distribution(generator) / FP(10);
        p(tag::Vel(), tag::Y()) = distribution(generator) / FP(10);
        p(tag::Vel(), tag::Z()) = distribution(generator) / FP(10);
        p(tag::Mass()) = distribution(generator) / FP(100);
        host_view(i) = p;
    }

    watch.printAndReset("init");

    alpaka::memcpy(queue, acc_buffer, host_buffer, buffer_size);
    watch.printAndReset("copy H->D");

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{static_cast<Size>(problem_size / (threads_per_block * desired_elements_per_thread))},
        alpaka::Vec<Dim, Size>{static_cast<Size>(threads_per_block)},
        alpaka::Vec<Dim, Size>{static_cast<Size>(desired_elements_per_thread)}};

    double sum_update = 0;
    double sum_move = 0;
    for (std::size_t s = 0; s < steps; ++s)
    {
        auto update_kernel = UpdateKernel<problem_size, desired_elements_per_thread, threads_per_block, TMappingSm>{};
        alpaka::exec<Acc>(queue, workdiv, update_kernel, acc_view);
        sum_update += watch.printAndReset("update", '\t');

        auto move_kernel = MoveKernel<problem_size, desired_elements_per_thread>{};
        alpaka::exec<Acc>(queue, workdiv, move_kernel, acc_view);
        sum_move += watch.printAndReset("move");
    }
    plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

    alpaka::memcpy(queue, host_buffer, acc_buffer, buffer_size);
    watch.printAndReset("copy D->H");
}

int main()
try
{
    std::cout << problem_size / 1000 << "k particles (" << problem_size * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << threads_per_block << " particles (" << threads_per_block * llama::sizeOf<Particle> / 1024
              << " kiB) in shared memory\n"
              << "Reducing on " << desired_elements_per_thread << " particles per thread\n"
              << "Using " << threads_per_block << " threads per block\n";
    std::cout << std::fixed;

    std::ofstream plot_file{"nbody.sh"};
    plot_file.exceptions(std::ios::badbit | std::ios::failbit);
    std::ofstream{"nbody.sh"} << R"(#!/usr/bin/gnuplot -p
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
$data << EOD
)";
    plot_file << "\"\"\t\"update\"\t\"move\"\n";

    // using Acc = alpaka::ExampleDefaultAcc;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;
    // using Acc = alpaka::AccCpuOmp2Blocks<Dim, Size>;

    run<alpaka::ExampleDefaultAcc, ao_s, ao_s>(plot_file);
    run<alpaka::ExampleDefaultAcc, ao_s, so_a>(plot_file);
    run<alpaka::ExampleDefaultAcc, ao_s, ao_so_a>(plot_file);
    run<alpaka::ExampleDefaultAcc, so_a, ao_s>(plot_file);
    run<alpaka::ExampleDefaultAcc, so_a, so_a>(plot_file);
    run<alpaka::ExampleDefaultAcc, so_a, ao_so_a>(plot_file);
    run<alpaka::ExampleDefaultAcc, ao_so_a, ao_s>(plot_file);
    run<alpaka::ExampleDefaultAcc, ao_so_a, so_a>(plot_file);
    run<alpaka::ExampleDefaultAcc, ao_so_a, ao_so_a>(plot_file);

    plot_file << R"(EOD
plot $data using 2:xtic(1) ti col
)";
    std::cout << "Plot with: ./nbody.sh\n";

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
