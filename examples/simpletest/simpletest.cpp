/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/// Simple example for LLAMA showing how to define array and record dimensions, to create a view and to access the data.

#include <iostream>
#include <llama/llama.hpp>
#include <sstream>
#include <utility>
#include <vector>

/// LLAMA uses class names for accessing data in the record dimension. It makes
/// sense to encapsulate these in own namespaces, especially if you are using
/// similar namings in differnt record dimensions, e.g. X, Y and Z in different
/// kinds of spaces.

// clang-format off
namespace st
{
    struct Pos{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Momentum{};
    struct Weight{};
    struct Options{};
} // namespace st

/// A record dimension in LLAMA is a type, probably always a \ref llama::Record. This takes a template list of all
/// members of this struct-like dimension. Every member needs to be a \ref llama::Field. A Field is a list of two
/// elements itself, first the name of the element as type and secondly the element type itself, which may be a nested
/// Record.
using Name = llama::Record<
    llama::Field<st::Pos, llama::Record<
        llama::Field<st::X, float>,
        llama::Field<st::Y, float>,
        llama::Field<st::Z, float>>>,
    llama::Field<st::Momentum, llama::Record<
        llama::Field<st::Z, double>,
        llama::Field<st::X, double>>>,
    llama::Field<st::Weight, int>,
    llama::Field<st::Options, bool[4]>>;
// clang-format on

namespace
{
    template <class T>
    auto type(const T& t) -> std::string
    {
        return boost::core::demangle(typeid(t).name());
    }

    /// Prints the coordinates of a given \ref llama::RecordCoord for debugging
    /// and testing purposes
    template <std::size_t... TRecordCoords>
    void print_coords(llama::RecordCoord<TRecordCoords...>)
    {
        (std::cout << ... << TRecordCoords);
    }

    template <typename TOut>
    void split(const std::string& s, char delim, TOut result)
    {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim))
        {
            *(result++) = item;
        }
    }

    auto split(const std::string& s, char delim) -> std::vector<std::string>
    {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }

    auto n_spaces(int n) -> std::string
    {
        std::string result;
        for (int i = 0; i < n; ++i)
            result += " ";
        return result;
    }

    auto add_line_breaks(std::string raw) -> std::string
    {
        using llama::mapping::tree::internal::replace_all;
        replace_all(raw, "<", "<\n");
        replace_all(raw, ", ", ",\n");
        replace_all(raw, " >", ">");
        replace_all(raw, ">", "\n>");
        auto tokens = split(raw, '\n');
        std::string result;
        int indent = 0;
        for (auto t : tokens)
        {
            if (t.back() == '>' || (t.length() > 1 && t[t.length() - 2] == '>'))
                indent -= 4;
            result += n_spaces(indent) + t + "\n";
            if (t.back() == '<')
                indent += 4;
        }
        return result;
    }
} // namespace

/// Example functor for \ref llama::forEachLeaf which can also be used to print the
/// coordinates inside of a record dimension when called.
template <typename TVirtualRecord>
struct SetZeroFunctor
{
    template <typename TCoord>
    void operator()(TCoord coord)
    {
        vd(coord) = 0;
    }
    TVirtualRecord vd;
};

auto main() -> int
try
{
    // Defining two array dimensions
    using ArrayDims = llama::ArrayDims<2>;
    // Setting the run time size of the array dimensions to 8192 * 8192
    ArrayDims ad_size{8192, 8192};

    // Printing dimensions information at runtime
    std::cout << "Record dimension is " << add_line_breaks(type(Name())) << '\n';
    std::cout << "AoS address of (0,100) <0,1>: "
              << llama::mapping::AoS<ArrayDims, Name>(ad_size).blobNrAndOffset<0, 1>({0, 100}).offset << '\n';
    std::cout << "SoA address of (0,100) <0,1>: "
              << llama::mapping::SoA<ArrayDims, Name>(ad_size).blobNrAndOffset<0, 1>({0, 100}).offset << '\n';
    std::cout << "sizeOf RecordDim: " << llama::sizeOf<Name> << '\n';

    std::cout << type(llama::GetCoordFromTags<Name, st::Pos, st::X>()) << '\n';

    // chosing a native struct of array mapping for this simple test example
    using Mapping = llama::mapping::SoA<ArrayDims, Name>;

    // Instantiating the mapping with the array dimensions size
    Mapping mapping(ad_size);
    // getting a view with memory from the default allocator
    auto view = allocView(mapping);

    // defining a position in the array dimensions
    const ArrayDims pos{0, 0};

    st::Options options;
    const auto weight = st::Weight{};

    // using the position in the array dimensions and a tree coord or a uid in the
    // record dimension to get the reference to an element in the view
    float& position_x = view(pos)(llama::RecordCoord<0, 0>{});
    double& momentum_z = view[pos](st::Momentum{}, st::Z{});
    int& weight = view[{0, 0}](llama::RecordCoord<2>{});
    int& weight_2 = view(pos)(weight);
    bool& options_2 = view[{0, 0}](st::Options())(llama::RecordCoord<2>());
    bool& options_3 = view(pos)(options)(llama::RecordCoord<2>());
    // printing the address and distances of the element in the memory. This
    // will change based on the chosen mapping. When array of struct is chosen
    // instead the elements will be much closer than with struct of array.
    std::cout << &position_x << '\n';
    std::cout << &momentum_z << " "
              << reinterpret_cast<std::byte*>(&momentum_z) - reinterpret_cast<std::byte*>(&position_x) << '\n';
    std::cout << &weight << " " << reinterpret_cast<std::byte*>(&weight) - reinterpret_cast<std::byte*>(&momentum_z)
              << '\n';
    std::cout << &options_2 << " " << reinterpret_cast<std::byte*>(&options_2) - reinterpret_cast<std::byte*>(&weight)
              << '\n';

    // iterating over the array dimensions at run time to do some stuff with the allocated data
    for (size_t x = 0; x < ad_size[0]; ++x)
        // telling the compiler that all data in the following loop is
        // independent to each other and thus can be vectorized
        LLAMA_INDEPENDENT_DATA
    for (size_t y = 0; y < ad_size[1]; ++y)
    {
        // Defining a functor for a given virtual record
        SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
        // Applying the functor for the sub tree 0,0 (pos.x), so basically
        // only for this element
        llama::forEachLeaf<Name>(szf, llama::RecordCoord<0, 0>{});
        // Applying the functor for the sub tree momentum (0), so basically
        // for momentum.z, and momentum.x
        llama::forEachLeaf<Name>(szf, st::Momentum{});
        // the array dimensions can be given as multiple comma separated arguments or as one parameter of type ArrayDims
        view({x, y}) = double(x + y) / double(ad_size[0] + ad_size[1]);
    }

    for (size_t x = 0; x < ad_size[0]; ++x)
        LLAMA_INDEPENDENT_DATA
    for (size_t y = 0; y < ad_size[1]; ++y)
    {
        // Showing different options of access data with llama. Internally
        // all do the same data- and mappingwise
        auto record = view(x, y);
        record(st::Pos{}, st::X{}) += static_cast<float>(record(llama::RecordCoord<1, 0>{}));
        record(st::Pos{}, st::Y{}) += static_cast<float>(record(llama::RecordCoord<1, 1>{}));
        record(st::Pos{}, st::Z{}) += static_cast<float>(record(llama::RecordCoord<2>{}));

        // It is also possible to work only on a part of data.
        record(st::Pos{}) += record(st::Momentum{});
    }
    double sum = 0.0;
    LLAMA_INDEPENDENT_DATA
    for (size_t x = 0; x < ad_size[0]; ++x)
        LLAMA_INDEPENDENT_DATA
    for (size_t y = 0; y < ad_size[1]; ++y)
        sum += view(x, y)(llama::RecordCoord<1, 0>{});
    std::cout << "Sum: " << sum << '\n';

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
