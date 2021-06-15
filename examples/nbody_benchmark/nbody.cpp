#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"

#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

constexpr auto problem_size = 16 * 1024;
constexpr auto steps = 10;
constexpr auto print_block_placement = false;

using FP = float;
constexpr FP timestep = 0.0001f;
constexpr FP ep_s2 = 0.01f;

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
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>
    >>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>
    >>,
    llama::Field<tag::Mass, FP>
>;
// clang-format on

template <typename TVirtualParticle>
LLAMA_FN_HOST_ACC_INLINE void p_p_interaction(TVirtualParticle p1, TVirtualParticle p2)
{
    auto dist = p1(tag::Pos{}) - p2(tag::Pos{});
    dist *= dist;
    const FP dist_sqr = ep_s2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const FP dist_sixth = dist_sqr * dist_sqr * dist_sqr;
    const FP inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
    const FP s = p2(tag::Mass{}) * inv_dist_cube;
    dist *= s * timestep;
    p1(tag::Vel{}) += dist;
}

template <typename TView>
void update(TView& particles)
{
    for (std::size_t i = 0; i < problem_size; i++)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t j = 0; j < problem_size; j++)
            p_p_interaction(particles(j), particles(i));
    }
}

template <typename TView>
void move(TView& particles)
{
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < problem_size; i++)
        particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * timestep;
}

template <std::size_t TMapping, std::size_t TAlignment>
void run(std::ostream& plot_file)
{
    std::cout << (TMapping == 0 ? "AoS" : TMapping == 1 ? "SoA" : "SoA MB") << ' ' << TAlignment << "\n";

    constexpr FP ts = 0.0001f;

    auto mapping = [&]
    {
        const auto array_dims = llama::ArrayDims{problem_size};
        if constexpr (TMapping == 0)
            return llama::mapping::AoS{array_dims, Particle{}};
        if constexpr (TMapping == 1)
            return llama::mapping::SoA{array_dims, Particle{}};
        if constexpr (TMapping == 2)
            return llama::mapping::SoA<decltype(array_dims), Particle, true>{array_dims};
    }();

    auto particles = llama::allocView(std::move(mapping), llama::bloballoc::Vector<TAlignment>{});

    if constexpr (print_block_placement)
    {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> blob_ranges;
        for (const auto& blob : particles.storageBlobs)
        {
            const auto blob_size = mapping.blobSize(blob_ranges.size());
            std::cout << "\tBlob #" << blob_ranges.size() << " from " << &blob[0] << " to " << &blob[0] + blob_size
                      << '\n';
            const auto start = reinterpret_cast<std::uint64_t>(&blob[0]);
            blob_ranges.emplace_back(start, start + blob_size);
        }
        std::sort(begin(blob_ranges), end(blob_ranges), [](auto a, auto b) { return a.first < b.first; });
        std::cout << "\tDistances: ";
        for (auto i = 0; i < blob_ranges.size() - 1; i++)
            std::cout << blob_ranges[i + 1].first - blob_ranges[i].first << ' ';
        std::cout << '\n';
        std::cout << "\tGaps: ";
        for (auto i = 0; i < blob_ranges.size() - 1; i++)
            std::cout << blob_ranges[i + 1].first - blob_ranges[i].second << ' ';
        std::cout << '\n';
    }

    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP(0), FP(1));
    for (std::size_t i = 0; i < problem_size; ++i)
    {
        auto p = particles(i);
        p(tag::Pos{}, tag::X{}) = dist(engine);
        p(tag::Pos{}, tag::Y{}) = dist(engine);
        p(tag::Pos{}, tag::Z{}) = dist(engine);
        p(tag::Vel{}, tag::X{}) = dist(engine) / FP(10);
        p(tag::Vel{}, tag::Y{}) = dist(engine) / FP(10);
        p(tag::Vel{}, tag::Z{}) = dist(engine) / FP(10);
        p(tag::Mass{}) = dist(engine) / FP(100);
    }

    double sum_update = 0;
    Stopwatch watch;
    for (std::size_t s = 0; s < steps; ++s)
    {
        update(particles);
        sum_update += watch.printAndReset("update", '\t');
        move(particles);
        watch.printAndReset("move");
    }

    if (TMapping == 0)
        plot_file << TAlignment;
    plot_file << '\t' << sum_update / steps << (TMapping == 2 ? '\n' : '\t');
}

auto main() -> int
try
{
    using namespace boost::mp11;

    std::ofstream plot_file{"nbody.sh"};
    plot_file.exceptions(std::ios::badbit | std::ios::failbit);
    plot_file << "\"alignment\"\t\"AoS\"\t\"SoA\"\t\"SoA MB\"\n";
    plot_file << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "nbody CPU {0}k particles on {1}"
set style data lines
set xtics rotate by 90 right
set key out top center maxrows 3
set yrange [0:*]
$data << EOD
)",
        problem_size / 1000,
        common::hostname());

    mp_for_each<mp_iota_c<28>>(
        [plot_file](auto ae)
        {
            mp_for_each<mp_list_c<std::size_t, 0, 1, 2>>(
                [plot_file](auto m)
                {
                    constexpr auto mapping = decltype(m)::value;
                    constexpr auto alignment = std::size_t{1} << decltype(ae)::value;
                    run<mapping, alignment>(plot_file);
                });
        });

    plot_file <<
        R"(EOD
plot $data using 2:xtic(1) ti col, '' using 3:xtic(1) ti col, '' using 4:xtic(1) ti col
)";
    std::cout << "Plot with: ./nbody.sh\n";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
