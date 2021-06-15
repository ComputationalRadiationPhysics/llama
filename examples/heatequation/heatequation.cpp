/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude,
 *                Sergei Bastrakov, Bernhard Manfred Gruber
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <llama/llama.hpp>
#include <utility>

template <typename TView>
inline void kernel(uint32_t idx, const TView& u_curr, TView& u_next, uint32_t extent, double dx, double dt)
{
    const auto r = dt / (dx * dx);
    if (idx > 0 && idx < extent - 1u)
        u_next[idx] = u_curr[idx] * (1.0 - 2.0 * r) + u_curr[idx - 1] * r + u_curr[idx + 1] * r;
}

template <typename TView>
void update_scalar(const TView& u_curr, TView& u_next, uint32_t extent, double dx, double dt)
{
    for (auto i = 0; i < extent; i++)
        kernel(i, u_curr, u_next, extent, dx, dt);
}

#if __has_include(<Vc/Vc>)
#    include <Vc/Vc>

template <typename TView>
inline void kernel_vec(uint32_t block_idx, const TView& u_curr, TView& u_next, uint32_t extent, double dx, double dt)
{
    const auto r = dt / (dx * dx);

    const auto base_idx = static_cast<uint32_t>(block_idx * Vc::double_v::size());
    if (base_idx > 0 && base_idx + Vc::double_v::size() < extent)
    {
        const auto next = Vc::double_v{&u_curr[base_idx]} * (1.0 - 2.0 * r) + Vc::double_v{&u_curr[base_idx - 1]} * r
            + Vc::double_v{&u_curr[base_idx + 1]} * r;
        next.store(&u_next[base_idx]);
    }
    else
    {
        for (auto idx = base_idx; idx <= base_idx + Vc::double_v::size(); idx++)
            if (idx > 0 && idx < extent - 1u)
                u_next[idx] = u_curr[idx] * (1.0 - 2.0 * r) + u_curr[idx - 1] * r + u_curr[idx + 1] * r;
    }
}

template <typename TView>
void update_vc(const TView& u_curr, TView& u_next, uint32_t extent, double dx, double dt)
{
    constexpr auto l = Vc::double_v::size();
    const auto blocks = (extent + l - 1) / l;
    for (auto i = 0; i < blocks; i++)
        kernel_vec(i, u_curr, u_next, extent, dx, dt);
}

template <typename TView>
void update_vc_peel(const TView& u_curr, TView& u_next, uint32_t extent, double dx, double dt)
{
    constexpr auto l = Vc::double_v::size();
    const auto blocks = (extent + l - 1) / l;
    kernel_vec(0, u_curr, u_next, extent, dx, dt);
    for (auto i = 1; i < blocks - 1; i++)
        kernel_vec(i, u_curr, u_next, extent, dx, dt);
    kernel_vec(blocks - 1, u_curr, u_next, extent, dx, dt);
}

#endif

// Exact solution to the test problem
// u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
// u(0, t) = u(1, t) = 0
// u(x, 0) = sin(pi * x)
auto exact_solution(double const x, double const t) -> double
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-pi * pi * t) * std::sin(pi * x);
}

auto main() -> int
try
{
    // Parameters (a user is supposed to change extent, timeSteps)
    const auto extent = 10000;
    const auto time_steps = 200000;
    const auto tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    const auto dx = 1.0 / static_cast<double>(extent - 1);
    const auto dt = tMax / static_cast<double>(time_steps - 1);

    const auto r = dt / (dx * dx);
    if (r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r << ", it is required to be <= 0.5\n";
        return 1;
    }

    const auto mapping = llama::mapping::SoA{llama::ArrayDims{extent}, double{}};
    auto uNext = llama::allocView(mapping);
    auto uCurr = llama::allocView(mapping);

    auto run = [&](std::string_view update_name, auto update)
    {
        // init
        for (uint32_t i = 0; i < extent; i++)
            uCurr[i] = exact_solution(i * dx, 0.0);

        // run simulation
        const auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < time_steps; step++)
        {
            update(uCurr, uNext, extent, dx, dt);
            std::swap(uNext, uCurr);
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << update_name << " took " << std::chrono::duration<double>(stop - start).count() << "s\t";

        // calculate error
        double max_error = 0.0;
        for (uint32_t i = 0; i < extent; i++)
        {
            const auto error = std::abs(uNext[i] - exact_solution(i * dx, tMax));
            max_error = std::max(max_error, error);
        }

        const auto error_threshold = 1e-5;
        const auto result_correct = (max_error < error_threshold);
        if (result_correct)
            std::cout << "Correct!\n";
        else
            std::cout << "Incorrect! error = " << max_error << " (the grid resolution may be too low)\n";
    };

    run("update_scalar ", [](auto&... args) { update_scalar(args...); });
#if __has_include(<Vc/Vc>)
    run("update_Vc     ", [](auto&... args) { update_vc(args...); });
    run("update_Vc_peel", [](auto&... args) { update_vc_peel(args...); });
#endif

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
