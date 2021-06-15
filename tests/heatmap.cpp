#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/llama.hpp>

TEST_CASE("Heatmap.nbody")
{
    constexpr auto n = 100;
    auto run = [&](const std::string& name, auto mapping)
    {
        auto particles = llama::allocView(llama::mapping::Heatmap{mapping});

        for (std::size_t i = 0; i < n; i++)
            particles(i) = 0;

        constexpr float timestep = 0.0001f;
        constexpr float ep_s2 = 0.01f;
        for (std::size_t i = 0; i < n; i++)
        {
            llama::One<ParticleHeatmap> pi = particles(i);
            for (std::size_t j = 0; j < n; ++j)
            {
                auto pj = particles(j);
                auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
                dist *= dist;
                const float dist_sqr = ep_s2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
                const float dist_sixth = dist_sqr * dist_sqr * dist_sqr;
                const float inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
                const float sts = pj(tag::Mass{}) * inv_dist_cube * timestep;
                pi(tag::Vel{}) += dist * sts;
            }
            particles(i) = pi;
        }
        for (std::size_t i = 0; i < n; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * timestep;

        std::ofstream{"Heatmap." + name + ".sh"} << particles.mapping.toGnuplotScript();
    };

    using ArrayDims = llama::ArrayDims<1>;
    auto array_dims = ArrayDims{n};
    run("AlignedAoS", llama::mapping::AlignedAoS<ArrayDims, ParticleHeatmap>{array_dims});
    run("SingleBlobSoA", llama::mapping::SingleBlobSoA<ArrayDims, ParticleHeatmap>{array_dims});
}
