#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"

#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <omp.h>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// needs -fno-math-errno, so std::sqrt() can be vectorized
// for multithreading, specify thread affinity (GNU OpenMP):
// e.g. for a 32 core CPU with SMT/hyperthreading: GOMP_CPU_AFFINITY='0-30:2,1-31:2' llama-nbody
// e.g. for a 16 core CPU without SMT/hyperthreading: GOMP_CPU_AFFINITY='0-15' llama-nbody

using FP = float;

constexpr auto problem_size = 16 * 1024;
constexpr auto steps = 5;
constexpr auto trace = false;
constexpr auto heatmap = false;
constexpr auto dump_mapping = false;
constexpr auto allow_rsqrt = true; // rsqrt can be way faster, but less accurate
constexpr auto newton_raphson_after_rsqrt
    = true; // generate a newton raphson refinement after explicit calls to rsqrt()
constexpr auto run_upate = true; // run update step. Useful to disable for benchmarking the move step.
constexpr FP timestep = 0.0001f;
constexpr FP ep_s2 = 0.01f;

constexpr auto l1_d_size = 32 * 1024;
constexpr auto l2_d_size = 512 * 1024;

using namespace std::string_literals;

namespace usellama
{
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

    template <typename TVirtualParticleI, typename TVirtualParticleJ>
    LLAMA_FN_HOST_ACC_INLINE void p_p_interaction(TVirtualParticleI& pi, TVirtualParticleJ pj)
    {
        auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
        dist *= dist;
        const FP dist_sqr = ep_s2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
        const FP dist_sixth = dist_sqr * dist_sqr * dist_sqr;
        const FP inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
        const FP sts = pj(tag::Mass{}) * inv_dist_cube * timestep;
        pi(tag::Vel{}) += dist * sts;
    }

    template <typename TView>
    void update(TView& particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            llama::One<Particle> pi = particles(i);
            for (std::size_t j = 0; j < problem_size; ++j)
                p_p_interaction(pi, particles(j));
            particles(i)(tag::Vel{}) = pi(tag::Vel{});
        }
    }

    template <typename TView>
    void move(TView& particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * timestep;
    }

    template <int TMapping, std::size_t TAoSoALanes = 8 /*AVX2*/>
    auto main(std::ostream& plot_file) -> int
    {
        auto mapping_name = [](int m) -> std::string
        {
            if (m == 0)
                return "AoS";
            if (m == 1)
                return "SoA";
            if (m == 2)
                return "SoA MB";
            if (m == 3)
                return "AoSoA" + std::to_string(TAoSoALanes);
            if (m == 4)
                return "Split SoA";
            std::abort();
        };
        auto title = "LLAMA " + mapping_name(TMapping);
        std::cout << title << "\n";
        Stopwatch watch;
        auto mapping = [&]
        {
            const auto array_dims = llama::ArrayDims{problem_size};
            if constexpr (TMapping == 0)
                return llama::mapping::AoS{array_dims, Particle{}};
            if constexpr (TMapping == 1)
                return llama::mapping::SoA{array_dims, Particle{}};
            if constexpr (TMapping == 2)
                return llama::mapping::SoA<decltype(array_dims), Particle, true>{array_dims};
            if constexpr (TMapping == 3)
                return llama::mapping::AoSoA<decltype(array_dims), Particle, TAoSoALanes>{array_dims};
            if constexpr (TMapping == 4)
                return llama::mapping::Split<
                    decltype(array_dims),
                    Particle,
                    llama::RecordCoord<1>,
                    llama::mapping::PreconfiguredSoA<>::type,
                    llama::mapping::PreconfiguredSoA<>::type,
                    true>{array_dims};
        }();
        if constexpr (dump_mapping)
            std::ofstream(title + ".svg") << llama::toSvg(mapping);

        auto tmapping = [&]
        {
            if constexpr (trace)
                return llama::mapping::Trace{std::move(mapping)};
            else
                return std::move(mapping);
        }();

        auto hmapping = [&]
        {
            if constexpr (heatmap)
                return llama::mapping::Heatmap{std::move(tmapping)};
            else
                return std::move(tmapping);
        }();

        auto particles = llama::allocView(std::move(hmapping));
        watch.printAndReset("alloc");

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
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                update(particles);
                sum_update += watch.printAndReset("update", '\t');
            }
            move(particles);
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        if constexpr (heatmap)
            std::ofstream("nbody_heatmap_" + mapping_name(TMapping) + ".sh") << particles.mapping.toGnuplotScript();

        return 0;
    }
} // namespace usellama

namespace manual_ao_s
{
    struct Vec
    {
        FP x;
        FP y;
        FP z;

        auto operator*=(FP s) -> Vec&
        {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        auto operator*=(Vec v) -> Vec&
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }

        auto operator+=(Vec v) -> Vec&
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        auto operator-=(Vec v) -> Vec&
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        friend auto operator-(Vec a, Vec b) -> Vec
        {
            return a -= b;
        }

        friend auto operator*(Vec a, FP s) -> Vec
        {
            return a *= s;
        }

        friend auto operator*(Vec a, Vec b) -> Vec
        {
            return a *= b;
        }
    };

    using Pos = Vec;
    using Vel = Vec;

    struct Particle
    {
        Pos pos;
        Vel vel;
        FP mass;
    };

    inline void p_p_interaction(Particle& pi, const Particle& pj)
    {
        auto distance = pi.pos - pj.pos;
        distance *= distance;
        const FP dist_sqr = ep_s2 + distance.x + distance.y + distance.z;
        const FP dist_sixth = dist_sqr * dist_sqr * dist_sqr;
        const FP inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
        const FP sts = pj.mass * inv_dist_cube * timestep;
        pi.vel += distance * sts;
    }

    void update(Particle* particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            Particle pi = particles[i];
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < problem_size; ++j)
                p_p_interaction(pi, particles[j]);
            particles[i].vel = pi.vel;
        }
    }

    void move(Particle* particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
            particles[i].pos += particles[i].vel * timestep;
    }

    auto main(std::ostream& plot_file) -> int
    {
        auto title = "AoS"s;
        std::cout << title << "\n";
        Stopwatch watch;

        std::vector<Particle> particles(problem_size);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (auto& p : particles)
        {
            p.pos.x = dist(engine);
            p.pos.y = dist(engine);
            p.pos.z = dist(engine);
            p.vel.x = dist(engine) / FP(10);
            p.vel.y = dist(engine) / FP(10);
            p.vel.z = dist(engine) / FP(10);
            p.mass = dist(engine) / FP(100);
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                update(particles.data());
                sum_update += watch.printAndReset("update", '\t');
            }
            move(particles.data());
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualAoS

namespace manual_so_a
{
    inline void p_p_interaction(
        FP piposx,
        FP piposy,
        FP piposz,
        FP& pivelx,
        FP& pively,
        FP& pivelz,
        FP pjposx,
        FP pjposy,
        FP pjposz,
        FP pjmass)
    {
        auto xdistance = piposx - pjposx;
        auto ydistance = piposy - pjposy;
        auto zdistance = piposz - pjposz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP dist_sqr = ep_s2 + xdistance + ydistance + zdistance;
        const FP dist_sixth = dist_sqr * dist_sqr * dist_sqr;
        const FP inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
        const FP sts = pjmass * inv_dist_cube * timestep;
        pivelx += xdistance * sts;
        pively += ydistance * sts;
        pivelz += zdistance * sts;
    }

    void update(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            const FP piposx = posx[i];
            const FP piposy = posy[i];
            const FP piposz = posz[i];
            FP pivelx = velx[i];
            FP pively = vely[i];
            FP pivelz = velz[i];
            for (std::size_t j = 0; j < problem_size; ++j)
                p_p_interaction(piposx, piposy, piposz, pivelx, pively, pivelz, posx[j], posy[j], posz[j], mass[j]);
            velx[i] = pivelx;
            vely[i] = pively;
            velz[i] = pivelz;
        }
    }

    void move(FP* posx, FP* posy, FP* posz, const FP* velx, const FP* vely, const FP* velz)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            posx[i] += velx[i] * timestep;
            posy[i] += vely[i] * timestep;
            posz[i] += velz[i] * timestep;
        }
    }

    auto main(std::ostream& plot_file) -> int
    {
        auto title = "SoA"s;
        std::cout << title << "\n";
        Stopwatch watch;

        using Vector = std::vector<FP, llama::bloballoc::AlignedAllocator<FP, 64>>;
        Vector posx(problem_size);
        Vector posy(problem_size);
        Vector posz(problem_size);
        Vector velx(problem_size);
        Vector vely(problem_size);
        Vector velz(problem_size);
        Vector mass(problem_size);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (std::size_t i = 0; i < problem_size; ++i)
        {
            posx[i] = dist(engine);
            posy[i] = dist(engine);
            posz[i] = dist(engine);
            velx[i] = dist(engine) / FP(10);
            vely[i] = dist(engine) / FP(10);
            velz[i] = dist(engine) / FP(10);
            mass[i] = dist(engine) / FP(100);
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                update(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data());
                sum_update += watch.printAndReset("update", '\t');
            }
            move(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data());
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualSoA

namespace manual_ao_so_a
{
    template <std::size_t TLanes>
    struct alignas(64) ParticleBlock
    {
        struct
        {
            FP x[TLanes];
            FP y[TLanes];
            FP z[TLanes];
        } pos;
        struct
        {
            FP x[TLanes];
            FP y[TLanes];
            FP z[TLanes];
        } vel;
        FP mass[TLanes];
    };

    inline void p_p_interaction(
        FP piposx,
        FP piposy,
        FP piposz,
        FP& pivelx,
        FP& pively,
        FP& pivelz,
        FP pjposx,
        FP pjposy,
        FP pjposz,
        FP pjmass)
    {
        auto xdistance = piposx - pjposx;
        auto ydistance = piposy - pjposy;
        auto zdistance = piposz - pjposz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP dist_sqr = ep_s2 + xdistance + ydistance + zdistance;
        const FP dist_sixth = dist_sqr * dist_sqr * dist_sqr;
        const FP inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
        const FP sts = pjmass * inv_dist_cube * timestep;
        pivelx += xdistance * sts;
        pively += ydistance * sts;
        pivelz += zdistance * sts;
    }

    template <std::size_t TLanes>
    void update(ParticleBlock<TLanes>* particles)
    {
        constexpr auto blocks = problem_size / TLanes;
        for (std::size_t bi = 0; bi < blocks; bi++)
        {
            auto block_i = particles[bi];
            for (std::size_t bj = 0; bj < blocks; bj++)
                for (std::size_t j = 0; j < TLanes; j++)
                {
                    LLAMA_INDEPENDENT_DATA
                    for (std::size_t i = 0; i < TLanes; i++)
                    {
                        const auto& block_j = particles[bj];
                        p_p_interaction(
                            block_i.pos.x[i],
                            block_i.pos.y[i],
                            block_i.pos.z[i],
                            block_i.vel.x[i],
                            block_i.vel.y[i],
                            block_i.vel.z[i],
                            block_j.pos.x[j],
                            block_j.pos.y[j],
                            block_j.pos.z[j],
                            block_j.mass[j]);
                    }
                }

            particles[bi].vel = block_i.vel;
        }
    }

    template <std::size_t TLanes>
    void update_tiled(ParticleBlock<TLanes>* particles)
    {
        constexpr auto blocks = problem_size / TLanes;
        constexpr auto blocks_per_tile = 128; // L1D_SIZE / sizeof(ParticleBlock<Lanes>);
        static_assert(blocks % blocks_per_tile == 0);
        for (std::size_t ti = 0; ti < blocks / blocks_per_tile; ti++)
            for (std::size_t tj = 0; tj < blocks / blocks_per_tile; tj++)
                for (std::size_t bi = 0; bi < blocks_per_tile; bi++)
                {
                    auto block_i = particles[ti * blocks_per_tile + bi];
                    for (std::size_t bj = 0; bj < blocks_per_tile; bj++)
                        for (std::size_t j = 0; j < TLanes; j++)
                        {
                            LLAMA_INDEPENDENT_DATA
                            for (std::size_t i = 0; i < TLanes; i++)
                            {
                                const auto& block_j = particles[tj * blocks_per_tile + bj];
                                p_p_interaction(
                                    block_i.pos.x[i],
                                    block_i.pos.y[i],
                                    block_i.pos.z[i],
                                    block_i.vel.x[i],
                                    block_i.vel.y[i],
                                    block_i.vel.z[i],
                                    block_j.pos.x[j],
                                    block_j.pos.y[j],
                                    block_j.pos.z[j],
                                    block_j.mass[j]);
                            }
                        }
                    particles[bi].vel = block_i.vel;
                }
    }

    template <std::size_t TLanes>
    void move(ParticleBlock<TLanes>* particles)
    {
        constexpr auto blocks = problem_size / TLanes;
        for (std::size_t bi = 0; bi < blocks; bi++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < TLanes; ++i)
            {
                auto& block = particles[bi];
                block.pos.x[i] += block.vel.x[i] * timestep;
                block.pos.y[i] += block.vel.y[i] * timestep;
                block.pos.z[i] += block.vel.z[i] * timestep;
            }
        }
    }

    template <std::size_t TLanes>
    auto main(std::ostream& plot_file, bool tiled) -> int
    {
        auto title = "AoSoA" + std::to_string(TLanes);
        if (tiled)
            title += " tiled";
        std::cout << title << "\n";
        Stopwatch watch;

        constexpr auto blocks = problem_size / TLanes;

        std::vector<ParticleBlock<TLanes>> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for (std::size_t i = 0; i < TLanes; ++i)
            {
                block.pos.x[i] = dist(engine);
                block.pos.y[i] = dist(engine);
                block.pos.z[i] = dist(engine);
                block.vel.x[i] = dist(engine) / FP(10);
                block.vel.y[i] = dist(engine) / FP(10);
                block.vel.z[i] = dist(engine) / FP(10);
                block.mass[i] = dist(engine) / FP(100);
            }
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                if (tiled)
                    update_tiled(particles.data());
                else
                    update(particles.data());
                sum_update += watch.printAndReset("update", '\t');
            }
            move(particles.data());
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualAoSoA

#ifdef __AVX2__
#    include <immintrin.h>

namespace manual_ao_so_a_manual_avx
{
    // hard coded to AVX2 register length, should be 8
    constexpr auto lanes = sizeof(__m256) / sizeof(float);

    struct alignas(32) ParticleBlock
    {
        struct
        {
            float x[lanes];
            float y[lanes];
            float z[lanes];
        } pos;
        struct
        {
            float x[lanes];
            float y[lanes];
            float z[lanes];
        } vel;
        float mass[lanes];
    };

    constexpr auto blocks = problem_size / lanes;
    const __m256 v_ep_s2 = _mm256_set1_ps(ep_s2); // NOLINT(cert-err58-cpp)
    const __m256 v_timestep = _mm256_set1_ps(timestep); // NOLINT(cert-err58-cpp)

    inline void p_p_interaction(
        __m256 piposx,
        __m256 piposy,
        __m256 piposz,
        __m256& pivelx,
        __m256& pively,
        __m256& pivelz,
        __m256 pjposx,
        __m256 pjposy,
        __m256 pjposz,
        __m256 pjmass)
    {
        const __m256 xdistance = _mm256_sub_ps(piposx, pjposx);
        const __m256 ydistance = _mm256_sub_ps(piposy, pjposy);
        const __m256 zdistance = _mm256_sub_ps(piposz, pjposz);
        const __m256 xdistance_sqr = _mm256_mul_ps(xdistance, xdistance);
        const __m256 ydistance_sqr = _mm256_mul_ps(ydistance, ydistance);
        const __m256 zdistance_sqr = _mm256_mul_ps(zdistance, zdistance);
        const __m256 dist_sqr
            = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v_ep_s2, xdistance_sqr), ydistance_sqr), zdistance_sqr);
        const __m256 dist_sixth = _mm256_mul_ps(_mm256_mul_ps(dist_sqr, dist_sqr), dist_sqr);
        const __m256 inv_dist_cube = [dist_sixth]
        {
            if constexpr (allow_rsqrt)
            {
                const __m256 r = _mm256_rsqrt_ps(dist_sixth);
                if constexpr (newton_raphson_after_rsqrt)
                {
                    // from: http://stackoverflow.com/q/14752399/556899
                    const __m256 three = _mm256_set1_ps(3.0f);
                    const __m256 half = _mm256_set1_ps(0.5f);
                    const __m256 muls = _mm256_mul_ps(_mm256_mul_ps(dist_sixth, r), r);
                    return _mm256_mul_ps(_mm256_mul_ps(half, r), _mm256_sub_ps(three, muls));
                }
                else
                    return r;
            }
            else
                return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(dist_sixth));
        }();
        const __m256 sts = _mm256_mul_ps(_mm256_mul_ps(pjmass, inv_dist_cube), v_timestep);
        pivelx = _mm256_fmadd_ps(xdistance_sqr, sts, pivelx);
        pively = _mm256_fmadd_ps(ydistance_sqr, sts, pively);
        pivelz = _mm256_fmadd_ps(zdistance_sqr, sts, pivelz);
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J
    void update8(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < blocks; bi++)
        {
            auto& block_i = particles[bi];
            const __m256 piposx = _mm256_load_ps(&block_i.pos.x[0]);
            const __m256 piposy = _mm256_load_ps(&block_i.pos.y[0]);
            const __m256 piposz = _mm256_load_ps(&block_i.pos.z[0]);
            __m256 pivelx = _mm256_load_ps(&block_i.vel.x[0]);
            __m256 pively = _mm256_load_ps(&block_i.vel.y[0]);
            __m256 pivelz = _mm256_load_ps(&block_i.vel.z[0]);

            for (std::size_t bj = 0; bj < blocks; bj++)
                for (std::size_t j = 0; j < lanes; j++)
                {
                    const auto& block_j = particles[bj];
                    const __m256 posx_j = _mm256_broadcast_ss(&block_j.pos.x[j]);
                    const __m256 posy_j = _mm256_broadcast_ss(&block_j.pos.y[j]);
                    const __m256 posz_j = _mm256_broadcast_ss(&block_j.pos.z[j]);
                    const __m256 mass_j = _mm256_broadcast_ss(&block_j.mass[j]);
                    p_p_interaction(piposx, piposy, piposz, pivelx, pively, pivelz, posx_j, posy_j, posz_j, mass_j);
                }

            _mm256_store_ps(&block_i.vel.x[0], pivelx);
            _mm256_store_ps(&block_i.vel.y[0], pively);
            _mm256_store_ps(&block_i.vel.z[0], pivelz);
        }
    }

    inline auto horizontal_sum(__m256 v) -> float
    {
        // from:
        // http://jtdz-solenoids.com/stackoverflow_/questions/13879609/horizontal-sum-of-8-packed-32bit-floats/18616679#18616679
        const __m256 t1 = _mm256_hadd_ps(v, v);
        const __m256 t2 = _mm256_hadd_ps(t1, t1);
        const __m128 t3 = _mm256_extractf128_ps(t2, 1); // NOLINT(hicpp-use-auto, modernize-use-auto)
        const __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
        return _mm_cvtss_f32(t4);

        // alignas(32) float a[LANES];
        //_mm256_store_ps(a, v);
        // return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    }

    // update (read/write) 1 particles I based on the influence of 8 particles J with accumulator
    void update1(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < blocks; bi++)
            for (std::size_t i = 0; i < lanes; i++)
            {
                auto& block_i = particles[bi];
                const __m256 piposx = _mm256_broadcast_ss(&block_i.pos.x[i]);
                const __m256 piposy = _mm256_broadcast_ss(&block_i.pos.y[i]);
                const __m256 piposz = _mm256_broadcast_ss(&block_i.pos.z[i]);
                __m256 pivelx = _mm256_broadcast_ss(&block_i.vel.x[i]);
                __m256 pively = _mm256_broadcast_ss(&block_i.vel.y[i]);
                __m256 pivelz = _mm256_broadcast_ss(&block_i.vel.z[i]);

                for (std::size_t bj = 0; bj < blocks; bj++)
                {
                    const auto& block_j = particles[bj];
                    const __m256 pjposx = _mm256_load_ps(&block_j.pos.x[0]);
                    const __m256 pjposy = _mm256_load_ps(&block_j.pos.y[0]);
                    const __m256 pjposz = _mm256_load_ps(&block_j.pos.z[0]);
                    const __m256 pjmass = _mm256_load_ps(&block_j.mass[0]);
                    p_p_interaction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
                }

                block_i.vel.x[i] = horizontal_sum(pivelx);
                block_i.vel.y[i] = horizontal_sum(pively);
                block_i.vel.z[i] = horizontal_sum(pivelz);
            }
    }

    void move(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < blocks; bi++)
        {
            auto& block = particles[bi];
            _mm256_store_ps(
                &block.pos.x[0],
                _mm256_fmadd_ps(_mm256_load_ps(&block.vel.x[0]), v_timestep, _mm256_load_ps(&block.pos.x[0])));
            _mm256_store_ps(
                &block.pos.y[0],
                _mm256_fmadd_ps(_mm256_load_ps(&block.vel.y[0]), v_timestep, _mm256_load_ps(&block.pos.y[0])));
            _mm256_store_ps(
                &block.pos.z[0],
                _mm256_fmadd_ps(_mm256_load_ps(&block.vel.z[0]), v_timestep, _mm256_load_ps(&block.pos.z[0])));
        }
    }

    auto main(std::ostream& plot_file, bool use_update1) -> int
    {
        auto title = "AoSoA" + std::to_string(lanes) + " AVX2 " + (use_update1 ? "w1r8" : "w8r1"); // NOLINT
        std::cout << title << '\n';
        Stopwatch watch;

        std::vector<ParticleBlock> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for (std::size_t i = 0; i < lanes; ++i)
            {
                block.pos.x[i] = dist(engine);
                block.pos.y[i] = dist(engine);
                block.pos.z[i] = dist(engine);
                block.vel.x[i] = dist(engine) / FP(10);
                block.vel.y[i] = dist(engine) / FP(10);
                block.vel.z[i] = dist(engine) / FP(10);
                block.mass[i] = dist(engine) / FP(100);
            }
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                if (use_update1)
                    update1(particles.data());
                else
                    update8(particles.data());
                sum_update += watch.printAndReset("update", '\t');
            }
            move(particles.data());
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualAoSoA_manualAVX
#endif

#if __has_include(<Vc/Vc>)
#    include <Vc/Vc>

namespace manual_ao_so_a_vc
{
    template <typename TVec>
    struct alignas(32) ParticleBlock
    {
        struct
        {
            TVec x;
            TVec y;
            TVec z;
        } pos, vel;
        TVec mass;
    };


    template <typename TVec>
    inline void p_p_interaction(
        TVec piposx,
        TVec piposy,
        TVec piposz,
        TVec& pivelx,
        TVec& pively,
        TVec& pivelz,
        TVec pjposx,
        TVec pjposy,
        TVec pjposz,
        TVec pjmass)
    {
        const TVec xdistance = piposx - pjposx;
        const TVec ydistance = piposy - pjposy;
        const TVec zdistance = piposz - pjposz;
        const TVec xdistance_sqr = xdistance * xdistance;
        const TVec ydistance_sqr = ydistance * ydistance;
        const TVec zdistance_sqr = zdistance * zdistance;
        const TVec dist_sqr = ep_s2 + xdistance_sqr + ydistance_sqr + zdistance_sqr;
        const TVec dist_sixth = dist_sqr * dist_sqr * dist_sqr;
        const TVec inv_dist_cube = [dist_sixth]
        {
            if constexpr (allow_rsqrt)
            {
                const TVec r = Vc::rsqrt(dist_sixth);
                if constexpr (newton_raphson_after_rsqrt)
                {
                    // from: http://stackoverflow.com/q/14752399/556899
                    const TVec three = 3.0f;
                    const TVec half = 0.5f;
                    const TVec muls = dist_sixth * r * r;
                    return (half * r) * (three - muls);
                }
                else
                    return r;
            }
            else
                return 1.0f / Vc::sqrt(dist_sixth);
        }();
        const TVec sts = pjmass * inv_dist_cube * timestep;
        pivelx = xdistance_sqr * sts + pivelx;
        pively = ydistance_sqr * sts + pively;
        pivelz = zdistance_sqr * sts + pivelz;
    }

    template <typename TVec>
    void update8(ParticleBlock<TVec>* particles, int threads)
    {
        constexpr auto lanes = TVec::size();
        constexpr auto blocks = problem_size / lanes;

#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < blocks; bi++)
        {
            auto& block_i = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            const TVec piposx = block_i.pos.x;
            const TVec piposy = block_i.pos.y;
            const TVec piposz = block_i.pos.z;
            TVec pivelx = block_i.vel.x;
            TVec pively = block_i.vel.y;
            TVec pivelz = block_i.vel.z;

            for (std::size_t bj = 0; bj < blocks; bj++)
                for (std::size_t j = 0; j < lanes; j++)
                {
                    const auto& block_j = particles[bj];
                    const TVec pjposx = block_j.pos.x[j];
                    const TVec pjposy = block_j.pos.y[j];
                    const TVec pjposz = block_j.pos.z[j];
                    const TVec pjmass = block_j.mass[j];

                    p_p_interaction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
                }

            block_i.vel.x = pivelx;
            block_i.vel.y = pively;
            block_i.vel.z = pivelz;
            // });
        }
    }

    template <typename TVec>
    void update8_tiled(ParticleBlock<TVec>* particles, int threads)
    {
        constexpr auto lanes = TVec::size();
        constexpr auto blocks = problem_size / lanes;

        constexpr auto blocks_per_tile = 128; // L1D_SIZE / sizeof(ParticleBlock);
        static_assert(blocks % blocks_per_tile == 0);
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t ti = 0; ti < blocks / blocks_per_tile; ti++)
            for (std::size_t bi = 0; bi < blocks_per_tile; bi++)
            {
                auto& block_i = particles[bi];
                const TVec piposx = block_i.pos.x;
                const TVec piposy = block_i.pos.y;
                const TVec piposz = block_i.pos.z;
                TVec pivelx = block_i.vel.x;
                TVec pively = block_i.vel.y;
                TVec pivelz = block_i.vel.z;
                for (std::size_t tj = 0; tj < blocks / blocks_per_tile; tj++)
                    for (std::size_t bj = 0; bj < blocks_per_tile; bj++)
                        for (std::size_t j = 0; j < lanes; j++)
                        {
                            const auto& block_j = particles[bj];
                            const TVec pjposx = block_j.pos.x[j];
                            const TVec pjposy = block_j.pos.y[j];
                            const TVec pjposz = block_j.pos.z[j];
                            const TVec pjmass = block_j.mass[j];

                            p_p_interaction(
                                piposx,
                                piposy,
                                piposz,
                                pivelx,
                                pively,
                                pivelz,
                                pjposx,
                                pjposy,
                                pjposz,
                                pjmass);
                        }

                block_i.vel.x = pivelx;
                block_i.vel.y = pively;
                block_i.vel.z = pivelz;
            }
    }

    template <typename TVec>
    void update1(ParticleBlock<TVec>* particles, int threads)
    {
        constexpr auto lanes = TVec::size();
        constexpr auto blocks = problem_size / lanes;

#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < blocks; bi++)
        {
            auto& block_i = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            for (std::size_t i = 0; i < lanes; i++)
            {
                const TVec piposx = static_cast<FP>(block_i.pos.x[i]);
                const TVec piposy = static_cast<FP>(block_i.pos.y[i]);
                const TVec piposz = static_cast<FP>(block_i.pos.z[i]);
                TVec pivelx = static_cast<FP>(block_i.vel.x[i]);
                TVec pively = static_cast<FP>(block_i.vel.y[i]);
                TVec pivelz = static_cast<FP>(block_i.vel.z[i]);

                for (std::size_t bj = 0; bj < blocks; bj++)
                {
                    const auto& block_j = particles[bj];
                    p_p_interaction(
                        piposx,
                        piposy,
                        piposz,
                        pivelx,
                        pively,
                        pivelz,
                        block_j.pos.x,
                        block_j.pos.y,
                        block_j.pos.z,
                        block_j.mass);
                }

                block_i.vel.x[i] = pivelx.sum();
                block_i.vel.y[i] = pively.sum();
                block_i.vel.z[i] = pivelz.sum();
            }
            // });
        }
    }

    template <typename TVec>
    void move(ParticleBlock<TVec>* particles, int threads)
    {
        constexpr auto blocks = problem_size / TVec::size();

#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < blocks; bi++)
        {
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& block) {
            auto& block = particles[bi];
            block.pos.x += block.vel.x * timestep;
            block.pos.y += block.vel.y * timestep;
            block.pos.z += block.vel.z * timestep;
            // });
        }
    }

    template <typename TVec>
    auto main(std::ostream& plot_file, int threads, bool use_update1, bool tiled = false) -> int
    {
        auto title = "AoSoA" + std::to_string(TVec::size()) + " Vc" + (use_update1 ? " w1r8" : " w8r1"); // NOLINT
        if (tiled)
            title += " tiled";
        if (threads > 1)
            title += " " + std::to_string(threads) + "Thrds";

        std::cout << title << '\n';
        Stopwatch watch;

        static_assert(problem_size % TVec::size() == 0);
        constexpr auto blocks = problem_size / TVec::size();
        std::vector<ParticleBlock<TVec>> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for (std::size_t i = 0; i < TVec::size(); ++i)
            {
                block.pos.x[i] = dist(engine);
                block.pos.y[i] = dist(engine);
                block.pos.z[i] = dist(engine);
                block.vel.x[i] = dist(engine) / FP(10);
                block.vel.y[i] = dist(engine) / FP(10);
                block.vel.z[i] = dist(engine) / FP(10);
                block.mass[i] = dist(engine) / FP(100);
            }
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                if (use_update1)
                    update1(particles.data(), threads);
                else
                {
                    if (tiled)
                        update8_tiled(particles.data(), threads);
                    else
                        update8(particles.data(), threads);
                }
                sum_update += watch.printAndReset("update", '\t');
            }
            move(particles.data(), threads);
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualAoSoA_Vc

namespace manual_ao_s_vc
{
    using manual_ao_s::Particle;
    using manual_ao_so_a_vc::p_p_interaction;

    template <typename TVec>
    const auto particle_gather_scatter_strides = 42;

    template <typename T>
    const auto particle_gather_scatter_strides<Vc::Vector<T>> = Vc::Vector<std::uint32_t>{Vc::IndexesFromZero}
        * std::uint32_t{sizeof(Particle) / sizeof(FP)};

    template <typename T, std::size_t N>
    const auto
        particle_gather_scatter_strides<Vc::SimdArray<T, N>> = Vc::SimdArray<std::uint32_t, N>{Vc::IndexesFromZero}
        * std::uint32_t{sizeof(Particle) / sizeof(FP)};

    template <typename TVec>
    void update(Particle* particles, int threads)
    {
        constexpr auto lanes = TVec::size();
        const auto strides = particle_gather_scatter_strides<TVec>;

#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t i = 0; i < problem_size; i += lanes)
        {
            // gather
            auto& pi = particles[i];
            const TVec piposx = TVec(&pi.pos.x, strides);
            const TVec piposy = TVec(&pi.pos.y, strides);
            const TVec piposz = TVec(&pi.pos.z, strides);
            TVec pivelx = TVec(&pi.vel.x, strides);
            TVec pively = TVec(&pi.vel.y, strides);
            TVec pivelz = TVec(&pi.vel.z, strides);

            for (std::size_t j = 0; j < problem_size; j++)
            {
                const auto& pj = particles[j];
                const TVec pjposx = pj.pos.x;
                const TVec pjposy = pj.pos.y;
                const TVec pjposz = pj.pos.z;
                const TVec pjmass = pj.mass;

                p_p_interaction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
            }

            // scatter
            pivelx.scatter(&pi.vel.x, strides);
            pively.scatter(&pi.vel.y, strides);
            pivelz.scatter(&pi.vel.z, strides);
        }
    }

    template <typename TVec>
    void move(Particle* particles, int threads)
    {
        constexpr auto lanes = TVec::size();
        const auto strides = particle_gather_scatter_strides<TVec>;

#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t i = 0; i < problem_size; i += lanes)
        {
            auto& pi = particles[i];
            (TVec(&pi.pos.x, strides) + TVec(&pi.vel.x, strides) * timestep).scatter(&pi.pos.x, strides);
            (TVec(&pi.pos.y, strides) + TVec(&pi.vel.y, strides) * timestep).scatter(&pi.pos.y, strides);
            (TVec(&pi.pos.z, strides) + TVec(&pi.vel.z, strides) * timestep).scatter(&pi.pos.z, strides);
        }
    }

    template <typename TVec>
    auto main(std::ostream& plot_file, int threads) -> int
    {
        auto title = "AoS Vc"s;
        if (threads > 1)
            title += " " + std::to_string(threads) + "Thrds";
        std::cout << title << '\n';
        Stopwatch watch;

        std::vector<Particle> particles(problem_size);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (auto& p : particles)
        {
            p.pos.x = dist(engine);
            p.pos.y = dist(engine);
            p.pos.z = dist(engine);
            p.vel.x = dist(engine) / FP(10);
            p.vel.y = dist(engine) / FP(10);
            p.vel.z = dist(engine) / FP(10);
            p.mass = dist(engine) / FP(100);
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                update<TVec>(particles.data(), threads);
                sum_update += watch.printAndReset("update", '\t');
            }
            move<TVec>(particles.data(), threads);
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualAoS_Vc

namespace manual_so_a_vc
{
    using manual_ao_so_a_vc::p_p_interaction;

    template <typename TVec>
    void update(
        const FP* posx,
        const FP* posy,
        const FP* posz,
        FP* velx,
        FP* vely,
        FP* velz,
        const FP* mass,
        int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t i = 0; i < problem_size; i += TVec::size())
        {
            const TVec piposx = TVec(posx + i);
            const TVec piposy = TVec(posy + i);
            const TVec piposz = TVec(posz + i);
            TVec pivelx = TVec(velx + i);
            TVec pively = TVec(vely + i);
            TVec pivelz = TVec(velz + i);
            for (std::size_t j = 0; j < problem_size; ++j)
                p_p_interaction(
                    piposx,
                    piposy,
                    piposz,
                    pivelx,
                    pively,
                    pivelz,
                    TVec(posx[j]),
                    TVec(posy[j]),
                    TVec(posz[j]),
                    TVec(mass[j]));
            pivelx.store(velx + i);
            pively.store(vely + i);
            pivelz.store(velz + i);
        }
    }

    template <typename TVec>
    void move(FP* posx, FP* posy, FP* posz, const FP* velx, const FP* vely, const FP* velz, int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t i = 0; i < problem_size; i += TVec::size())
        {
            (TVec(posx + i) + TVec(velx + i) * timestep).store(posx + i);
            (TVec(posy + i) + TVec(vely + i) * timestep).store(posy + i);
            (TVec(posz + i) + TVec(velz + i) * timestep).store(posz + i);
        }
    }

    template <typename TVec>
    auto main(std::ostream& plot_file, int threads) -> int
    {
        auto title = "SoA Vc"s;
        if (threads > 1)
            title += " " + std::to_string(threads) + "Thrds";
        std::cout << title << '\n';
        Stopwatch watch;

        using Vector = std::vector<FP, llama::bloballoc::AlignedAllocator<FP, 64>>;
        Vector posx(problem_size);
        Vector posy(problem_size);
        Vector posz(problem_size);
        Vector velx(problem_size);
        Vector vely(problem_size);
        Vector velz(problem_size);
        Vector mass(problem_size);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (std::size_t i = 0; i < problem_size; ++i)
        {
            posx[i] = dist(engine);
            posy[i] = dist(engine);
            posz[i] = dist(engine);
            velx[i] = dist(engine) / FP(10);
            vely[i] = dist(engine) / FP(10);
            velz[i] = dist(engine) / FP(10);
            mass[i] = dist(engine) / FP(100);
        }
        watch.printAndReset("init");

        double sum_update = 0;
        double sum_move = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            if constexpr (run_upate)
            {
                update<TVec>(
                    posx.data(),
                    posy.data(),
                    posz.data(),
                    velx.data(),
                    vely.data(),
                    velz.data(),
                    mass.data(),
                    threads);
                sum_update += watch.printAndReset("update", '\t');
            }
            move<TVec>(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), threads);
            sum_move += watch.printAndReset("move");
        }
        plot_file << std::quoted(title) << "\t" << sum_update / steps << '\t' << sum_move / steps << '\n';

        return 0;
    }
} // namespace manualSoA_Vc
#endif

auto main() -> int
try
{
#if __has_include(<Vc/Vc>)
    using Vec = Vc::Vector<FP>;
    // using vec = Vc::SimdArray<FP, 16>;
    constexpr auto simd_lanes = vec::size();
#else
    constexpr auto SIMDLanes = 1;
#endif

    const auto num_threads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY");
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;

    fmt::print(
        R"({}ki particles ({}kiB)
Threads: {}
Affinity: {}
SIMD lanes: {}
)",
        problem_size / 1024,
        problem_size * sizeof(FP) * 7 / 1024,
        num_threads,
        affinity,
        simd_lanes);

    std::ofstream plot_file{"nbody.sh"};
    plot_file.exceptions(std::ios::badbit | std::ios::failbit);
    plot_file << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# threads: {} affinity: {} SIMD lanes: {}
set title "nbody CPU {}ki particles on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
set y2range [0:*]
set ylabel "update runtime [s]"
set y2label "move runtime [s]"
set y2tics auto
$data << EOD
)",
        num_threads,
        affinity,
        simd_lanes,
        problem_size / 1024,
        common::hostname());
    plot_file << "\"\"\t\"update\"\t\"move\"\n";

    // Note:
    // Tiled versions did not give any performance benefit, so they are disabled by default.
    // SIMD versions updating 8 particles by 1 are also a bit faster than updating 1 particle by 8, so the latter are
    // also disabled.

    int r = 0;
    using namespace boost::mp11;
    mp_for_each<mp_iota_c<5>>(
        [plot_file](auto i)
        {
            // only AoSoA (3) needs lanes
            using Lanes
                = std::conditional_t<decltype(i)::value == 3, mp_list_c<std::size_t, 8, 16>, mp_list_c<std::size_t, 0>>;
            mp_for_each<Lanes>([plot_file, i](auto lanes)
                               { r += usellama::main<decltype(i)::value, decltype(lanes)::value>(plot_file); });
        });
    r += manual_ao_s::main(plot_file);
    r += manual_so_a::main(plot_file);
    mp_for_each<mp_list_c<std::size_t, 8, 16>>(
        [plot_file](auto lanes)
        {
            // for (auto tiled : {false, true})
            //    r += manualAoSoA::main<decltype(lanes)::value>(plotFile, tiled);
            r += manual_ao_so_a::main<decltype(lanes)::value>(plot_file, false);
        });
#ifdef __AVX2__
    // for (auto useUpdate1 : {false, true})
    //    r += manualAoSoA_manualAVX::main(plotFile, useUpdate1);
    r += manual_ao_so_a_manual_avx::main(plot_file, false);
#endif
#if __has_include(<Vc/Vc>)
    for (int threads = 1; threads <= std::thread::hardware_concurrency(); threads *= 2)
    {
        // for (auto useUpdate1 : {false, true})
        //    for (auto tiled : {false, true})
        //    {
        //        if (useUpdate1 && tiled)
        //            continue;
        //        r += manualAoSoA_Vc::main<vec>(plotFile, threads, useUpdate1, tiled);
        //    }
        r += manual_ao_so_a_vc::main<vec>(plot_file, threads, false, false);
    }
    for (int threads = 1; threads <= std::thread::hardware_concurrency(); threads *= 2)
    {
        // mp_for_each<mp_list_c<std::size_t, 1, 2, 4, 8, 16>>(
        //    [&](auto lanes) { r += manualAoS_Vc::main<Vc::SimdArray<FP, decltype(lanes)::value>>(plotFile, threads);
        //    });
        r += manual_ao_s_vc::main<vec>(plot_file, threads);
    }
    for (int threads = 1; threads <= std::thread::hardware_concurrency(); threads *= 2)
        r += manual_so_a_vc::main<vec>(plot_file, threads);
#endif

    plot_file << R"(EOD
plot $data using 2:xtic(1) ti col axis x1y1, "" using 3 ti col axis x1y2
)";
    std::cout << "Plot with: ./nbody.sh\n";

    return r;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
