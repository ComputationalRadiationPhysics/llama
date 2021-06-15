#include <algorithm>
#include <array>
#include <cstring>
#include <fmt/core.h>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
    struct Z{};
} // namespace tag

using Vector = llama::Record<
    llama::Field<tag::X, int>,
    llama::Field<tag::Y, int>,
    llama::Field<tag::Z, int>
>;
// clang-format on

template <template <typename, typename> typename TInnerMapping, typename TTArrayDims, typename TTRecordDim>
struct GuardMapping2D
{
    static_assert(std::is_same_v<TTArrayDims, llama::ArrayDims<2>>, "Only 2D arrays are implemented");

    using ArrayDims = TTArrayDims;
    using RecordDim = TTRecordDim;

    constexpr GuardMapping2D() = default;

    constexpr explicit GuardMapping2D(ArrayDims size, RecordDim = {})
        : m_array_dims_size(size)
        , m_left({size[0] - 2})
        , m_right({size[0] - 2})
        , m_top({size[1] - 2})
        , m_bot({size[1] - 2})
        , m_center({size[0] - 2, size[1] - 2})
    {
    }

    constexpr auto array_dims() const -> ArrayDims
    {
        return m_array_dims_size;
    }

    constexpr auto blob_size(std::size_t i) const -> std::size_t
    {
        if (i >= center_off)
            return m_center.blobSize(i - center_off);
        if (i >= bot_off)
            return m_bot.blobSize(i - bot_off);
        if (i >= top_off)
            return m_top.blobSize(i - top_off);
        if (i >= right_off)
            return m_right.blobSize(i - right_off);
        if (i >= left_off)
            return m_left.blobSize(i - left_off);
        if (i >= right_bot_off)
            return m_right_bot.blobSize(i - right_bot_off);
        if (i >= right_top_off)
            return m_right_top.blobSize(i - right_top_off);
        if (i >= left_bot_off)
            return m_left_bot.blobSize(i - left_bot_off);
        if (i >= left_top_off)
            return m_left_top.blobSize(i - left_top_off);
        std::abort();
    }

    template <std::size_t... TRecordCoords>
    constexpr auto blob_nr_and_offset(ArrayDims coord) const -> llama::NrAndOffset
    {
        // [0][0] is at left top
        const auto [row, col] = coord;
        const auto [rowMax, colMax] = m_array_dims_size;

        if (col == 0)
        {
            if (row == 0)
                return offset_blob_nr(m_left_top.template blobNrAndOffset<TRecordCoords...>({}), left_top_off);
            if (row == rowMax - 1)
                return offset_blob_nr(m_left_bot.template blobNrAndOffset<TRecordCoords...>({}), left_bot_off);
            return offset_blob_nr(m_left.template blobNrAndOffset<TRecordCoords...>({row - 1}), left_off);
        }
        if (col == colMax - 1)
        {
            if (row == 0)
                return offset_blob_nr(m_right_top.template blobNrAndOffset<TRecordCoords...>({}), right_top_off);
            if (row == rowMax - 1)
                return offset_blob_nr(m_right_bot.template blobNrAndOffset<TRecordCoords...>({}), right_bot_off);
            return offset_blob_nr(m_right.template blobNrAndOffset<TRecordCoords...>({row - 1}), right_off);
        }
        if (row == 0)
            return offset_blob_nr(m_top.template blobNrAndOffset<TRecordCoords...>({col - 1}), top_off);
        if (row == rowMax - 1)
            return offset_blob_nr(m_bot.template blobNrAndOffset<TRecordCoords...>({col - 1}), bot_off);
        return offset_blob_nr(m_center.template blobNrAndOffset<TRecordCoords...>({row - 1, col - 1}), center_off);
    }

    constexpr auto center_blobs() const
    {
        return blob_indices(m_center, center_off);
    }

    constexpr auto left_top_blobs() const
    {
        return blob_indices(m_left_top, left_top_off);
    }

    constexpr auto left_bot_blobs() const
    {
        return blobIndices(m_left_bot, left_bot_off);
    }

    constexpr auto left_blobs() const
    {
        return blob_indices(m_left, left_off);
    }

    constexpr auto right_top_blobs() const
    {
        return blobIndices(m_right_top, right_top_off);
    }

    constexpr auto right_bot_blobs() const
    {
        return blob_indices(m_right_bot, right_bot_off);
    }

    constexpr auto right_blobs() const
    {
        return blob_indices(m_right, right_off);
    }

    constexpr auto top_blobs() const
    {
        return blobIndices(m_top, top_off);
    }

    constexpr auto bot_blobs() const
    {
        return blobIndices(m_bot, bot_off);
    }

private:
    constexpr auto offset_blob_nr(llama::NrAndOffset nao, std::size_t blob_nr_offset) const -> llama::NrAndOffset
    {
        nao.nr += blob_nr_offset;
        return nao;
    }

    template <typename TMapping>
    constexpr auto blob_indices(const TMapping&, std::size_t offset) const
    {
        std::array<std::size_t, TMapping::blob_count> a{};
        std::generate(begin(a), end(a), [i = offset]() mutable { return i++; });
        return a;
    }

    llama::mapping::One<ArrayDims, RecordDim> m_left_top;
    llama::mapping::One<ArrayDims, RecordDim> m_left_bot;
    llama::mapping::One<ArrayDims, RecordDim> m_right_top;
    llama::mapping::One<ArrayDims, RecordDim> m_right_bot;
    InnerMapping<llama::ArrayDims<1>, RecordDim> m_left;
    InnerMapping<llama::ArrayDims<1>, RecordDim> m_right;
    InnerMapping<llama::ArrayDims<1>, RecordDim> m_top;
    InnerMapping<llama::ArrayDims<1>, RecordDim> m_bot;
    InnerMapping<llama::ArrayDims<2>, RecordDim> m_center;

    static constexpr auto left_top_off = std::size_t{0};
    static constexpr auto left_bot_off = left_top_off + decltype(m_left_top)::blobCount;
    static constexpr auto right_top_off = left_bot_off + decltype(m_left_bot)::blobCount;
    static constexpr auto right_bot_off = right_top_off + decltype(m_right_top)::blobCount;
    static constexpr auto left_off = right_bot_off + decltype(m_right_bot)::blobCount;
    static constexpr auto right_off = left_off + decltype(m_left)::blobCount;
    static constexpr auto top_off = right_off + decltype(m_right)::blobCount;
    static constexpr auto bot_off = top_off + decltype(m_top)::blobCount;
    static constexpr auto center_off = bot_off + decltype(m_bot)::blobCount;

public:
    static constexpr auto blob_count = center_off + decltype(m_center)::blobCount;

private:
    ArrayDims m_array_dims_size;
};

template <typename TView>
void print_view(const TView& view, std::size_t rows, std::size_t cols)
{
    for (std::size_t row = 0; row < rows; row++)
    {
        for (std::size_t col = 0; col < cols; col++)
            std::cout << fmt::format(
                "[{:3},{:3},{:3}]",
                view(row, col)(tag::X{}),
                view(row, col)(tag::Y{}),
                view(row, col)(tag::Z{}));
        std::cout << '\n';
    }
}

template <template <typename, typename> typename TMapping>
void run(const std::string& mapping_name)
{
    std::cout << "\n===== Mapping " << mapping_name << " =====\n\n";

    constexpr auto rows = 7;
    constexpr auto cols = 5;
    const auto array_dims = llama::ArrayDims{rows, cols};
    const auto mapping = GuardMapping2D<Mapping, llama::ArrayDims<2>, Vector>{array_dims};
    std::ofstream{"bufferguard_" + mapping_name + ".svg"} << llama::toSvg(mapping);

    auto view1 = allocView(mapping);

    int i = 0;
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
        {
            view1(row, col)(tag::X{}) = ++i;
            view1(row, col)(tag::Y{}) = ++i;
            view1(row, col)(tag::Z{}) = ++i;
        }

    std::cout << "View 1:\n";
    print_view(view1, rows, cols);

    auto view2 = allocView(mapping);
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            view2(row, col) = 0; // broadcast

    std::cout << "\nView 2:\n";
    print_view(view2, rows, cols);

    auto copy_blobs = [&](auto& src_view, auto& dst_view, auto src_blobs, auto dst_blobs)
    {
        static_assert(src_blobs.size() == dst_blobs.size());
        for (auto i = 0; i < src_blobs.size(); i++)
        {
            const auto src = src_blobs[i];
            const auto dst = dst_blobs[i];
            assert(mapping.blobSize(src) == mapping.blobSize(dst));
            std::memcpy(&dst_view.storageBlobs[dst][0], &src_view.storageBlobs[src][0], mapping.blob_size(src));
        }
    };

    std::cout << "\nCopy view 1 right -> view 2 left:\n";
    copy_blobs(view1, view2, mapping.right_blobs(), mapping.left_blobs());

    std::cout << "View 2:\n";
    print_view(view2, rows, cols);

    std::cout << "\nCopy view 1 left top -> view 2 right bot:\n";
    copy_blobs(view1, view2, mapping.left_top_blobs(), mapping.right_bot_blobs());

    std::cout << "View 2:\n";
    print_view(view2, rows, cols);

    std::cout << "\nCopy view 2 center -> view 1 center:\n";
    copy_blobs(view2, view1, mapping.center_blobs(), mapping.center_blobs());

    std::cout << "View 1:\n";
    print_view(view1, rows, cols);
}

auto main() -> int
try
{
    run<llama::mapping::PreconfiguredAoS<>::type>("AoS");
    run<llama::mapping::PreconfiguredSoA<>::type>("SoA");
    run<llama::mapping::PreconfiguredSoA<true>::type>("SoA_MB");
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
