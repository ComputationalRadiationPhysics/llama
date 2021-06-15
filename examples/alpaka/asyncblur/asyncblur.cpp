/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file asynccopy.cpp
 *  \brief Asynchronous bluring example for LLAMA using ALPAKA.
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../common/Stopwatch.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <list>
#include <llama/llama.hpp>
#include <random>
#include <stb_image.h>
#include <stb_image_write.h>
#include <utility>

constexpr auto async = true; ///< defines whether the data shall be processed asynchronously
constexpr auto shared = true; ///< defines whether shared memory shall be used
constexpr auto save = true; ///< defines whether the resultion image shall be saved
constexpr auto chunk_count = 4;

constexpr auto default_img_x = 4096; /// width of the default image if no png is loaded
constexpr auto default_img_y = 4096; /// height of the default image if no png is loaded
constexpr auto kernel_size = 8; /// radius of the blur kernel, the diameter is this times two plus one
constexpr auto chunk_size = 512; /// size of each chunk to be processed per alpaka kernel
constexpr auto elems_per_block = 16; /// number of elements per direction(!) every block should process

using FP = float;

template <typename TMapping, typename TAlpakaBuffer>
auto view_alpaka_buffer(
    TMapping& mapping,
    TAlpakaBuffer& buffer) // taking mapping by & on purpose, so Mapping can deduce const
{
    return llama::View<TMapping, std::byte*>{mapping, {alpaka::getPtrNative(buffer)}};
}

// clang-format off
namespace tag
{
    struct R{};
    struct G{};
    struct B{};
} // namespace tag

/// real record dimension of the image pixel used on the host for loading and saving
using Pixel = llama::Record<
    llama::Field<tag::R, FP>,
    llama::Field<tag::G, FP>,
    llama::Field<tag::B, FP>>;

/// record dimension used in the kernel to modify the image
using PixelOnAcc = llama::Record<
    llama::Field<tag::R, FP>, // you can remove one here if you want to checkout the difference of the result image ;)
    llama::Field<tag::G, FP>,
    llama::Field<tag::B, FP>>;
// clang-format on

/** Alpaka kernel functor used to blur a small image living in the device memory
 *  using the \ref PixelOnAcc record dimension
 */
template <std::size_t TElems, std::size_t TKernelSize, std::size_t TElemsPerBlock>
struct BlurKernel
{
    template <typename TAcc, typename TView>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const TAcc& acc, TView old_image, TView new_image) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        [[maybe_unused]] auto shared_view = [&]
        {
            if constexpr (shared)
            {
                // Using SoA for the shared memory
                constexpr auto shared_chunk_size = TElemsPerBlock + 2 * TKernelSize;
                const auto shared_mapping = llama::mapping::SoA(
                    typename TView::ArrayDims{shared_chunk_size, shared_chunk_size},
                    typename TView::RecordDim{});
                constexpr auto shared_mem_size = llama::sizeOf<PixelOnAcc> * shared_chunk_size * shared_chunk_size;
                auto& shared_mem = alpaka::declareSharedVar<std::byte[shared_mem_size], __COUNTER__>(acc);
                return llama::View(shared_mapping, llama::Array<std::byte*, 1>{&shared_mem[0]});
            }
            else
                return int{}; // dummy
        }();

        [[maybe_unused]] const auto bi = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        if constexpr (shared)
        {
            constexpr auto threads_per_block = TElemsPerBlock / TElems;
            const auto thread_idx_in_block = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

            const std::size_t b_start[2]
                = {bi[0] * TElemsPerBlock + thread_idx_in_block[0], bi[1] * TElemsPerBlock + thread_idx_in_block[1]};
            const std::size_t b_end[2] = {
                alpaka::math::min(
                    acc,
                    b_start[0] + TElemsPerBlock + 2 * TKernelSize,
                    old_image.mapping.arrayDimsSize[0]),
                alpaka::math::min(
                    acc,
                    b_start[1] + TElemsPerBlock + 2 * TKernelSize,
                    old_image.mapping.arrayDimsSize[1]),
            };
            LLAMA_INDEPENDENT_DATA
            for (auto y = b_start[0]; y < b_end[0]; y += threads_per_block)
                LLAMA_INDEPENDENT_DATA
            for (auto x = b_start[1]; x < b_end[1]; x += threads_per_block)
                shared_view(y - bi[0] * TElemsPerBlock, x - bi[1] * TElemsPerBlock) = old_image(y, x);

            alpaka::syncBlockThreads(acc);
        }

        const std::size_t start[2] = {ti[0] * TElems, ti[1] * TElems};
        const std::size_t end[2] = {
            alpaka::math::min(acc, start[0] + TElems, old_image.mapping.arrayDimsSize[0] - 2 * TKernelSize),
            alpaka::math::min(acc, start[1] + TElems, old_image.mapping.arrayDimsSize[1] - 2 * TKernelSize),
        };

        LLAMA_INDEPENDENT_DATA
        for (auto y = start[0]; y < end[0]; ++y)
            LLAMA_INDEPENDENT_DATA
        for (auto x = start[1]; x < end[1]; ++x)
        {
            llama::One<PixelOnAcc> sum{0};

            using ItType = std::int64_t;
            const ItType i_b_start = shared ? ItType(y) - ItType(bi[0] * TElemsPerBlock) : y;
            const ItType i_a_start = shared ? ItType(x) - ItType(bi[1] * TElemsPerBlock) : x;
            const ItType i_b_end
                = shared ? ItType(y + 2 * TKernelSize + 1) - ItType(bi[0] * TElemsPerBlock) : y + 2 * TKernelSize + 1;
            const ItType i_a_end
                = shared ? ItType(x + 2 * TKernelSize + 1) - ItType(bi[1] * TElemsPerBlock) : x + 2 * TKernelSize + 1;
            LLAMA_INDEPENDENT_DATA
            for (auto b = i_b_start; b < i_b_end; ++b)
                LLAMA_INDEPENDENT_DATA
            for (auto a = i_a_start; a < i_a_end; ++a)
            {
                if constexpr (shared)
                    sum += shared_view(std::size_t(b), std::size_t(a));
                else
                    sum += old_image(std::size_t(b), std::size_t(a));
            }
            sum /= FP((2 * TKernelSize + 1) * (2 * TKernelSize + 1));
            new_image(y + TKernelSize, x + TKernelSize) = sum;
        }
    }
};

auto main(int argc, char** argv) -> int
try
{
    // ALPAKA
    using Dim = alpaka::DimInt<2>;

    using Acc = alpaka::ExampleDefaultAcc<Dim, std::size_t>;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;

    using Queue = alpaka::Queue<Acc, std::conditional_t<async, alpaka::NonBlocking, alpaka::Blocking>>;
    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    const DevAcc dev_acc = alpaka::getDevByIdx<PltfAcc>(0);
    const DevHost dev_host = alpaka::getDevByIdx<PltfHost>(0);
    std::vector<Queue> queue;
    for (std::size_t i = 0; i < chunk_count; ++i)
        queue.emplace_back(dev_acc);

    // ASYNCCOPY
    std::size_t img_x = default_img_x;
    std::size_t img_y = default_img_y;
    std::size_t buffer_x = default_img_x + 2 * kernel_size;
    std::size_t buffer_y = default_img_y + 2 * kernel_size;

    constexpr std::size_t hardware_threads = 2; // relevant for OpenMP2Threads
    using Distribution = common::ThreadsElemsDistribution<Acc, elems_per_block, hardware_threads>;
    constexpr std::size_t elem_count = Distribution::elemCount;
    constexpr std::size_t thread_count = Distribution::threadCount;

    std::vector<unsigned char> image;
    std::string out_filename = "output.png";

    if (argc > 1)
    {
        int x = 0;
        int y = 0;
        int n = 3;
        unsigned char* data = stbi_load(argv[1], &x, &y, &n, 0);
        image.resize(x * y * 3);
        std::copy(data, data + image.size(), begin(image));
        stbi_image_free(data);
        img_x = x;
        img_y = y;
        buffer_x = x + 2 * kernel_size;
        buffer_y = y + 2 * kernel_size;

        if (argc > 2)
            out_filename = std::string(argv[2]);
    }

    // LLAMA
    using ArrayDims = llama::ArrayDims<2>;

    auto tree_operation_list = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
    const auto host_mapping
        = llama::mapping::tree::Mapping{ArrayDims{buffer_y, buffer_x}, tree_operation_list, Pixel{}};
    const auto dev_mapping = llama::mapping::tree::Mapping{
        ArrayDims{chunk_size + 2 * kernel_size, chunk_size + 2 * kernel_size},
        tree_operation_list,
        PixelOnAcc{}};

    const auto host_buffer_size = host_mapping.blobSize(0);
    const auto dev_buffer_size = dev_mapping.blobSize(0);

    std::cout << "Image size: " << img_x << ":" << img_y << '\n'
              << host_buffer_size * 2 / 1024 / 1024 << " MB on device\n";

    Stopwatch chrono;

    auto host_buffer = alpaka::allocBuf<std::byte, std::size_t>(dev_host, host_buffer_size);
    auto host_view = view_alpaka_buffer(host_mapping, host_buffer);

    std::vector<alpaka::Buf<DevHost, std::byte, alpaka::DimInt<1>, std::size_t>> host_chunk_buffer;
    std::vector<llama::View<decltype(dev_mapping), std::byte*>> host_chunk_view;

    std::vector<alpaka::Buf<DevAcc, std::byte, alpaka::DimInt<1>, std::size_t>> dev_old_buffer;
    std::vector<alpaka::Buf<DevAcc, std::byte, alpaka::DimInt<1>, std::size_t>> dev_new_buffer;
    std::vector<llama::View<decltype(dev_mapping), std::byte*>> dev_old_view;
    std::vector<llama::View<decltype(dev_mapping), std::byte*>> dev_new_view;

    for (std::size_t i = 0; i < chunk_count; ++i)
    {
        host_chunk_buffer.push_back(alpaka::allocBuf<std::byte, std::size_t>(dev_host, dev_buffer_size));
        host_chunk_view.push_back(view_alpaka_buffer(dev_mapping, host_chunk_buffer.back()));

        dev_old_buffer.push_back(alpaka::allocBuf<std::byte, std::size_t>(dev_acc, dev_buffer_size));
        dev_old_view.push_back(view_alpaka_buffer(dev_mapping, dev_old_buffer.back()));

        dev_new_buffer.push_back(alpaka::allocBuf<std::byte, std::size_t>(dev_acc, dev_buffer_size));
        dev_new_view.push_back(view_alpaka_buffer(dev_mapping, dev_new_buffer.back()));
    }

    chrono.printAndReset("Alloc");

    if (image.empty())
    {
        image.resize(img_x * img_y * 3);
        std::default_random_engine generator;
        std::normal_distribution<FP> distribution{FP(0), FP(0.5)};
        for (std::size_t y = 0; y < buffer_y; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t x = 0; x < buffer_x; ++x)
            {
                host_view(y, x)(tag::R()) = std::abs(distribution(generator));
                host_view(y, x)(tag::G()) = std::abs(distribution(generator));
                host_view(y, x)(tag::B()) = std::abs(distribution(generator));
            }
        }
    }
    else
    {
        for (std::size_t y = 0; y < buffer_y; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t x = 0; x < buffer_x; ++x)
            {
                const auto x = std::clamp<std::size_t>(x, kernel_size, img_x + kernel_size - 1);
                const auto y = std::clamp<std::size_t>(y, kernel_size, img_y + kernel_size - 1);
                const auto* pixel = &image[((y - kernel_size) * img_x + x - kernel_size) * 3];
                host_view(y, x)(tag::R()) = FP(pixel[0]) / 255;
                host_view(y, x)(tag::G()) = FP(pixel[1]) / 255;
                host_view(y, x)(tag::B()) = FP(pixel[2]) / 255;
            }
        }
    }

    chrono.printAndReset("Init");
    const auto elems = alpaka::Vec<Dim, size_t>(elem_count, elem_count);
    const auto threads = alpaka::Vec<Dim, size_t>(thread_count, thread_count);
    const auto blocks = alpaka::Vec<Dim, size_t>(
        static_cast<size_t>((chunk_size + elems_per_block - 1) / elems_per_block),
        static_cast<size_t>((chunk_size + elems_per_block - 1) / elems_per_block));
    const alpaka::Vec<Dim, size_t> chunks(
        static_cast<size_t>((img_y + chunk_size - 1) / chunk_size),
        static_cast<size_t>((img_x + chunk_size - 1) / chunk_size));

    const auto workdiv = alpaka::WorkDivMembers<Dim, size_t>{blocks, threads, elems};

    struct VirtualHostElement
    {
        llama::VirtualView<decltype(host_view)> virtual_host;
        const ArrayDims m_valid_mini_size;
    };
    std::list<VirtualHostElement> virtual_host_list;
    for (std::size_t chunk_y = 0; chunk_y < chunks[0]; ++chunk_y)
        for (std::size_t chunk_x = 0; chunk_x < chunks[1]; ++chunk_x)
        {
            // Create virtual view with size of mini view
            const ArrayDims valid_mini_size{
                ((chunk_y < chunks[0] - 1) ? chunk_size : (img_y - 1) % chunk_size + 1) + 2 * kernel_size,
                ((chunk_x < chunks[1] - 1) ? chunk_size : (img_x - 1) % chunk_size + 1) + 2 * kernel_size};
            llama::VirtualView virtual_host(host_view, {chunk_y * chunk_size, chunk_x * chunk_size});

            // Find free chunk stream
            std::size_t chunk_nr = virtual_host_list.size();
            if (virtual_host_list.size() < chunk_count)
                virtual_host_list.push_back({virtual_host, valid_mini_size});
            else
            {
                bool not_found = true;
                while (not_found)
                {
                    auto chunk_it = virtual_host_list.begin();
                    for (chunk_nr = 0; chunk_nr < chunk_count; ++chunk_nr)
                    {
                        if (alpaka::empty(queue[chunk_nr]))
                        {
                            // Copy data back
                            LLAMA_INDEPENDENT_DATA
                            for (std::size_t y = 0; y < chunk_it->m_valid_mini_size[0] - 2 * kernel_size; ++y)
                            {
                                LLAMA_INDEPENDENT_DATA
                                for (std::size_t x = 0; x < chunk_it->m_valid_mini_size[1] - 2 * kernel_size; ++x)
                                    chunk_it->virtual_host(y + kernel_size, x + kernel_size)
                                        = host_chunk_view[chunk_nr](y + kernel_size, x + kernel_size);
                            }
                            chunk_it = virtual_host_list.erase(chunk_it);
                            virtual_host_list.insert(chunk_it, {virtual_host, valid_mini_size});
                            not_found = false;
                            break;
                        }
                        chunk_it++;
                    }
                    if (not_found)
                        std::this_thread::sleep_for(std::chrono::microseconds{1});
                }
            }

            // Copy data from virtual view to mini view
            for (std::size_t y = 0; y < valid_mini_size[0]; ++y)
            {
                LLAMA_INDEPENDENT_DATA
                for (std::size_t x = 0; x < valid_mini_size[1]; ++x)
                    host_chunk_view[chunk_nr](y, x) = virtual_host(y, x);
            }
            alpaka::memcpy(queue[chunk_nr], dev_old_buffer[chunk_nr], host_chunk_buffer[chunk_nr], dev_buffer_size);

            alpaka::exec<Acc>(
                queue[chunk_nr],
                workdiv,
                BlurKernel<elem_count, kernel_size, elems_per_block>{},
                dev_old_view[chunk_nr],
                dev_new_view[chunk_nr]);

            alpaka::memcpy(queue[chunk_nr], host_chunk_buffer[chunk_nr], dev_new_buffer[chunk_nr], dev_buffer_size);
        }

    // Wait for not finished tasks on accelerator
    auto chunk_it = virtual_host_list.begin();
    for (std::size_t chunk_nr = 0; chunk_nr < chunk_count; ++chunk_nr)
    {
        alpaka::wait(queue[chunk_nr]);
        // Copy data back
        for (std::size_t y = 0; y < chunk_it->m_valid_mini_size[0] - 2 * kernel_size; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t x = 0; x < chunk_it->m_valid_mini_size[1] - 2 * kernel_size; ++x)
                chunk_it->virtual_host(y + kernel_size, x + kernel_size)
                    = host_chunk_view[chunk_nr](y + kernel_size, x + kernel_size);
        }
        chunk_it++;
    }
    chrono.printAndReset("Blur kernel");

    if (save)
    {
        for (std::size_t y = 0; y < img_y; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t x = 0; x < img_x; ++x)
            {
                auto* pixel = &image[(y * img_x + x) * 3];
                pixel[0] = host_view(y + kernel_size, x + kernel_size)(tag::R()) * 255.;
                pixel[1] = host_view(y + kernel_size, x + kernel_size)(tag::G()) * 255.;
                pixel[2] = host_view(y + kernel_size, x + kernel_size)(tag::B()) * 255.;
            }
        }
        stbi_write_png(out_filename.c_str(), img_x, img_y, 3, image.data(), 0);
    }

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
