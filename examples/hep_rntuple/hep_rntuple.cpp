// This example uses a non-public CMS NanoAOD file called: ttjet_13tev_june2019_lzma.
// Please ask contact us if you need it.

#include "../common/ttjet_13tev_june2019.hpp"

#include <RConfigure.h>
#define R__HAS_STD_STRING_VIEW
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleView.hxx>
#include <chrono>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        fmt::print("Please specify input file!\n");
        return 1;
    }

    using namespace std::chrono;
    using namespace ROOT::Experimental;

    auto ntuple = RNTupleReader::Open(RNTupleModel::Create(), "NTuple", argv[1]);
    const auto n = ntuple->GetNEntries();

    auto start = steady_clock::now();
    auto view = llama::allocView(llama::mapping::SoA<llama::ArrayDims<1>, Event, true>{llama::ArrayDims{n}});
    fmt::print("Alloc LLAMA view: {}ms\n", duration_cast<milliseconds>(steady_clock::now() - start).count());

    std::size_t totalSize = 0;
    for (auto i = 0u; i < view.mapping.blobCount; i++)
        totalSize += view.mapping.blobSize(i);
    fmt::print("Total LLAMA view memory: {}MiB in {} blobs\n", totalSize / 1024 / 1024, view.mapping.blobCount);

    start = steady_clock::now();
    llama::forEachLeaf<Event>(
        [&](auto coord)
        {
            using Name = llama::GetTag<Event, decltype(coord)>;
            using Type = llama::GetType<Event, decltype(coord)>;
            auto column = ntuple->GetView<Type>(llama::structName<Name>());
            for (std::size_t i = 0; i < n; i++)
                view(i)(coord) = column(i);
        });
    fmt::print("Copy RNTuple -> LLAMA view: {}ms\n", duration_cast<milliseconds>(steady_clock::now() - start).count());

    start = steady_clock::now();
}
