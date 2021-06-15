#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"

#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>

constexpr auto mapping = 2; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blobs), 3 tree AoS, 4 tree SoA
constexpr auto problem_size = 64 * 1024 * 1024; ///< problem size
constexpr auto steps = 10; ///< number of vector adds to perform

using FP = float;

namespace usellama
{
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
        llama::Field<tag::Z, FP>
    >;
    // clang-format on

    template <typename TView>
    void add(const TView& a, const TView& b, TView& c)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            c(i)(tag::X{}) = a(i)(tag::X{}) + b(i)(tag::X{});
            c(i)(tag::Y{}) = a(i)(tag::Y{}) - b(i)(tag::Y{});
            c(i)(tag::Z{}) = a(i)(tag::Z{}) * b(i)(tag::Z{});
        }
    }

    auto main(std::ofstream& plot_file) -> int
    {
        std::cout << "\nLLAMA\n";
        Stopwatch watch;

        const auto mapping = [&]
        {
            const auto array_dims = llama::ArrayDims{problem_size};
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

        auto a = allocView(mapping);
        auto b = allocView(mapping);
        auto c = allocView(mapping);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; ++i)
        {
            a[i](tag::X{}) = i; // X
            a[i](tag::Y{}) = i; // Y
            a[i](llama::RecordCoord<2>{}) = i; // Z
            b(i) = i; // writes to all (X, Y, Z)
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            add(a, b, c);
            acc += watch.printAndReset("add");
        }
        plot_file << "LLAMA\t" << acc / steps << '\n';

        return static_cast<int>(c.storageBlobs[0][0]);
    }
} // namespace usellama

namespace manual_ao_s
{
    struct Vector
    {
        FP x;
        FP y;
        FP z;
    };

    inline void add(const Vector* a, const Vector* b, Vector* c)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            c[i].x = a[i].x + b[i].x;
            c[i].y = a[i].y - b[i].y;
            c[i].z = a[i].z * b[i].z;
        }
    }

    auto main(std::ofstream& plot_file) -> int
    {
        std::cout << "\nAoS\n";
        Stopwatch watch;

        std::vector<Vector> a(problem_size);
        std::vector<Vector> b(problem_size);
        std::vector<Vector> c(problem_size);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; ++i)
        {
            a[i].x = i;
            a[i].y = i;
            a[i].z = i;
            b[i].x = i;
            b[i].y = i;
            b[i].z = i;
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            add(a.data(), b.data(), c.data());
            acc += watch.printAndReset("add");
        }
        plot_file << "AoS\t" << acc / steps << '\n';

        return c[0].x;
    }
} // namespace manualAoS

namespace manual_so_a
{
    inline void add(
        const FP* ax,
        const FP* ay,
        const FP* az,
        const FP* bx,
        const FP* by,
        const FP* bz,
        FP* cx,
        FP* cy,
        FP* cz)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; i++)
        {
            cx[i] = ax[i] + bx[i];
            cy[i] = ay[i] - by[i];
            cz[i] = az[i] * bz[i];
        }
    }

    auto main(std::ofstream& plot_file) -> int
    {
        std::cout << "\nSoA\n";
        Stopwatch watch;

        using Vector = std::vector<float, llama::bloballoc::AlignedAllocator<float, 64>>;
        Vector ax(problem_size);
        Vector ay(problem_size);
        Vector az(problem_size);
        Vector bx(problem_size);
        Vector by(problem_size);
        Vector bz(problem_size);
        Vector cx(problem_size);
        Vector cy(problem_size);
        Vector cz(problem_size);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < problem_size; ++i)
        {
            ax[i] = i;
            ay[i] = i;
            az[i] = i;
            bx[i] = i;
            by[i] = i;
            bz[i] = i;
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            add(ax.data(), ay.data(), az.data(), bx.data(), by.data(), bz.data(), cx.data(), cy.data(), cz.data());
            acc += watch.printAndReset("add");
        }
        plot_file << "SoA\t" << acc / steps << '\n';

        return cx[0];
    }
} // namespace manualSoA


namespace manual_ao_so_a
{
    constexpr auto lanes = 16;

    struct alignas(64) VectorBlock
    {
        FP x[lanes];
        FP y[lanes];
        FP z[lanes];
    };

    constexpr auto blocks = problem_size / lanes;

    inline void add(const VectorBlock* a, const VectorBlock* b, VectorBlock* c)
    {
        for (std::size_t bi = 0; bi < problem_size / lanes; bi++)
        {
// the unroll 1 is needed to prevent unrolling, which prevents vectorization in GCC
#pragma GCC unroll 1
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < lanes; ++i)
            {
                const auto& block_a = a[bi];
                const auto& block_b = b[bi];
                auto& block_c = c[bi];
                block_c.x[i] = block_a.x[i] + block_b.x[i];
                block_c.y[i] = block_a.y[i] - block_b.y[i];
                block_c.z[i] = block_a.z[i] * block_b.z[i];
            }
        }
    }

    auto main(std::ofstream& plot_file) -> int
    {
        std::cout << "\nAoSoA\n";
        Stopwatch watch;

        std::vector<VectorBlock> a(blocks);
        std::vector<VectorBlock> b(blocks);
        std::vector<VectorBlock> c(blocks);
        watch.printAndReset("alloc");

        for (std::size_t bi = 0; bi < problem_size / lanes; ++bi)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < lanes; ++i)
            {
                a[bi].x[i] = bi * lanes + i;
                a[bi].y[i] = bi * lanes + i;
                a[bi].z[i] = bi * lanes + i;
                b[bi].x[i] = bi * lanes + i;
                b[bi].y[i] = bi * lanes + i;
                b[bi].z[i] = bi * lanes + i;
            }
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < steps; ++s)
        {
            add(a.data(), b.data(), c.data());
            acc += watch.printAndReset("add");
        }
        plot_file << "AoSoA\t" << acc / steps << '\n';

        return c[0].x[0];
    }
} // namespace manualAoSoA


auto main() -> int
try
{
    std::cout << problem_size / 1000 / 1000 << "M values "
              << "(" << problem_size * sizeof(float) / 1024 << "kiB)\n";

    std::ofstream plot_file{"vectoradd.sh"};
    plot_file.exceptions(std::ios::badbit | std::ios::failbit);
    plot_file << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "vectoradd CPU {}Mi elements on {}"
set style data histograms
set style fill solid
#set key out top center maxrows 3
set yrange [0:*]
set ylabel "update runtime [s]"
$data << EOD
)",
        problem_size / 1024 / 1024,
        common::hostname());

    int r = 0;
    r += usellama::main(plot_file);
    r += manual_ao_s::main(plot_file);
    r += manual_so_a::main(plot_file);
    r += manual_ao_so_a::main(plot_file);

    plot_file << R"(EOD
plot $data using 2:xtic(1) ti "runtime"
)";
    std::cout << "Plot with: ./vectoradd.sh\n";

    return r;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
