#include <chrono>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

// needs -fno-math-errno, so std::sqrt() can be vectorized

constexpr auto MAPPING = 2; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blos), 3 tree AoS, 4 tree SoA
constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr auto TRACE = false;

using FP = float;
constexpr FP EPS2 = 0.01f;

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
    }

    using Particle = llama::DS<
        llama::DE<tag::Pos, llama::DS<
            llama::DE<tag::X, FP>,
            llama::DE<tag::Y, FP>,
            llama::DE<tag::Z, FP>
        >>,
        llama::DE<tag::Vel, llama::DS<
            llama::DE<tag::X, FP>,
            llama::DE<tag::Y, FP>,
            llama::DE<tag::Z, FP>
        >>,
        llama::DE<tag::Mass, FP>
    >;
    // clang-format on

    template <typename VirtualParticle>
    LLAMA_FN_HOST_ACC_INLINE void pPInteraction(VirtualParticle p1, VirtualParticle p2, FP ts)
    {
        auto dist = p1(tag::Pos{}) - p2(tag::Pos{});
        dist *= dist;
        const FP distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2(tag::Mass{}) * invDistCube;
        dist *= s * ts;
        p1(tag::Vel{}) += dist;
    }

    template <typename View>
    void update(View& particles, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(particles(j), particles(i), ts);
        }
    }

    template <typename View>
    void move(View& particles, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * ts;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
        auto mapping = [&] {
            if constexpr (MAPPING == 0)
                return llama::mapping::AoS{arrayDomain, Particle{}};
            if constexpr (MAPPING == 1)
                return llama::mapping::SoA{arrayDomain, Particle{}};
            if constexpr (MAPPING == 2)
                return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
            if constexpr (MAPPING == 3)
                return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Particle{}};
            if constexpr (MAPPING == 4)
                return llama::mapping::tree::Mapping{
                    arrayDomain,
                    llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                    Particle{}};
        }();

        auto tmapping = [&] {
            if constexpr (TRACE)
                return llama::mapping::Trace{std::move(mapping)};
            else
                return std::move(mapping);
        }();

        auto particles = llama::allocView(std::move(tmapping));

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles LLAMA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
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

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles, ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles, ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace usellama

namespace manualAoS
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

    inline void pPInteraction(Particle& p1, const Particle& p2, FP ts)
    {
        auto distance = p1.pos - p2.pos;
        distance *= distance;
        const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2.mass * invDistCube;
        distance *= s * ts;
        p1.vel += distance;
    }

    void update(Particle* particles, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(particles[j], particles[i], ts);
        }
    }

    void move(Particle* particles, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles[i].pos += particles[i].vel * ts;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles AoS\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<Particle> particles(PROBLEM_SIZE);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

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

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoS

namespace manualSoA
{
    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP& p1velx,
        FP& p1vely,
        FP& p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx - p2posx;
        auto ydistance = p1posy - p2posy;
        auto zdistance = p1posz - p2posz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2mass * invDistCube;
        xdistance *= s * ts;
        ydistance *= s * ts;
        zdistance *= s * ts;
        p1velx += xdistance;
        p1vely += ydistance;
        p1velz += zdistance;
    }

    void update(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(
                    posx[j],
                    posy[j],
                    posz[j],
                    velx[j],
                    vely[j],
                    velz[j],
                    posx[i],
                    posy[i],
                    posz[i],
                    mass[i],
                    ts);
        }
    }

    void move(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            posx[i] += velx[i] * ts;
            posy[i] += vely[i] * ts;
            posz[i] += velz[i] * ts;
        }
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles SoA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<FP> posx(PROBLEM_SIZE);
        std::vector<FP> posy(PROBLEM_SIZE);
        std::vector<FP> posz(PROBLEM_SIZE);
        std::vector<FP> velx(PROBLEM_SIZE);
        std::vector<FP> vely(PROBLEM_SIZE);
        std::vector<FP> velz(PROBLEM_SIZE);
        std::vector<FP> mass(PROBLEM_SIZE);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
            {
                posx[i] = dist(engine);
                posy[i] = dist(engine);
                posz[i] = dist(engine);
                velx[i] = dist(engine) / FP(10);
                vely[i] = dist(engine) / FP(10);
                velz[i] = dist(engine) / FP(10);
                mass[i] = dist(engine) / FP(100);
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualSoA

namespace manualAoSoA
{
    constexpr auto LANES = 16;

    struct ParticleBlock
    {
        struct
        {
            FP x[LANES];
            FP y[LANES];
            FP z[LANES];
        } pos;
        struct
        {
            FP x[LANES];
            FP y[LANES];
            FP z[LANES];
        } vel;
        FP mass[LANES];
    };

    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP& p1velx,
        FP& p1vely,
        FP& p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx - p2posx;
        auto ydistance = p1posy - p2posy;
        auto zdistance = p1posz - p2posz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2mass * invDistCube;
        xdistance *= s * ts;
        ydistance *= s * ts;
        zdistance *= s * ts;
        p1velx += xdistance;
        p1vely += ydistance;
        p1velz += zdistance;
    }

    void update(ParticleBlock* particles, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE / LANES; i++)
        {
            for (std::size_t j = 0; j < PROBLEM_SIZE / LANES; j++)
            {
                auto& blockI = particles[i];
                for (std::size_t ii = 0; ii < LANES; ii++)
                {
                    auto& blockJ = particles[j];
                    LLAMA_INDEPENDENT_DATA
                    for (std::size_t jj = 0; jj < LANES; jj++)
                    {
                        pPInteraction(
                            blockJ.pos.x[jj],
                            blockJ.pos.y[jj],
                            blockJ.pos.z[jj],
                            blockJ.vel.x[jj],
                            blockJ.vel.y[jj],
                            blockJ.vel.z[jj],
                            blockI.pos.x[ii],
                            blockI.pos.y[ii],
                            blockI.pos.z[ii],
                            blockI.mass[ii],
                            ts);
                    }
                }
            }
        }
    }

    void move(ParticleBlock* particles, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE / LANES; i++)
        {
            auto& block = particles[i];
            LLAMA_INDEPENDENT_DATA
            for (std::size_t ii = 0; ii < LANES; ++ii)
            {
                block.pos.x[ii] += block.vel.x[ii] * ts;
                block.pos.y[ii] += block.vel.y[ii] * ts;
                block.pos.z[ii] += block.vel.z[ii] * ts;
            }
        }
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles AoSoA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<ParticleBlock> particles(PROBLEM_SIZE / LANES);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (std::size_t i = 0; i < PROBLEM_SIZE / LANES; ++i)
            {
                auto& block = particles[i];
                for (std::size_t ii = 0; ii < LANES; ++ii)
                {
                    block.pos.x[ii] = dist(engine);
                    block.pos.y[ii] = dist(engine);
                    block.pos.z[ii] = dist(engine);
                    block.vel.x[ii] = dist(engine) / FP(10);
                    block.vel.y[ii] = dist(engine) / FP(10);
                    block.vel.z[ii] = dist(engine) / FP(10);
                    block.mass[ii] = dist(engine) / FP(100);
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoSoA

int main(int argc, char** argv)
{
    int r = 0;
    r += usellama::main(argc, argv);
    r += manualAoS::main(argc, argv);
    r += manualSoA::main(argc, argv);
    r += manualAoSoA::main(argc, argv);
    return r;
}
