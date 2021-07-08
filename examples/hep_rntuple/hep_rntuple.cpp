// This example uses a non-public CMS NanoAOD file called: ttjet_13tev_june2019_lzma.
// Please contact us if you need it.

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

using SmallEvent = boost::mp11::mp_take_c<Event, 100>;

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        fmt::print("Please specify input file!\n");
        return 1;
    }

    using namespace std::chrono;
    using namespace ROOT::Experimental;

    // auto ntuple
    //    = RNTupleReader::Open(RNTupleModel::Create(), "NTuple", "/mnt/c/dev/llama/ttjet_13tev_june2019_lzma.root");
    auto ntuple = RNTupleReader::Open(RNTupleModel::Create(), "NTuple", argv[1]);
    // try
    //{
    //    ntuple->PrintInfo(ROOT::Experimental::ENTupleInfo::kStorageDetails);
    //}
    // catch (const std::exception& e)
    //{
    //    fmt::print("PrintInfo error: {}", e.what());
    //}
    const auto eventCount = ntuple->GetNEntries();
    const auto& d = ntuple->GetDescriptor();
    const auto electronCount
        = d.GetNElements(d.FindColumnId(d.FindFieldId("nElectron.nElectron.Electron_deltaEtaSC"), 0));
    fmt::print("File contains {} events with {} electrons\n", eventCount, electronCount);

    auto start = steady_clock::now();
    auto mapping = llama::mapping::OffsetTable<llama::ArrayDims<1>, SmallEvent>{
        llama::ArrayDims{eventCount},
        llama::ArrayDims{electronCount}};
    auto view = llama::allocView(mapping);
    fmt::print("Alloc LLAMA view: {}ms\n", duration_cast<milliseconds>(steady_clock::now() - start).count());

    std::size_t totalSize = 0;
    for (auto i = 0u; i < view.mapping.blobCount; i++)
        totalSize += view.mapping.blobSize(i);
    fmt::print("Total LLAMA view memory: {}MiB in {} blobs\n", totalSize / 1024 / 1024, view.mapping.blobCount);

    // fill offset table
    start = steady_clock::now();
    std::size_t offset = 0;
    auto electronViewCollection = ntuple->GetViewCollection("nElectron");
    for (std::size_t i = 0; i < eventCount; i++)
    {
        offset += electronViewCollection(i);
        view(i)(llama::EndOffset<nElectron>{}) = offset;
        assert(offset <= electronCount);
    }
    fmt::print("Fill offset table: {}ms\n", duration_cast<milliseconds>(steady_clock::now() - start).count());

    using AugmentedSmallEvent = typename decltype(mapping)::RecordDim;
    start = steady_clock::now();
    llama::forEachLeaf<AugmentedSmallEvent>(
        [&](auto coord)
        {
            using Coord = decltype(coord);
            using LeafTag = llama::GetTag<AugmentedSmallEvent, Coord>;
            using Type = llama::GetType<AugmentedSmallEvent, Coord>;

            fmt::print("Copying {}\n", llama::structName<LeafTag>());
            if constexpr (
                !llama::mapping::internal::isEndOffsetField<LeafTag> && !llama::mapping::internal::isSizeField<LeafTag>)
            {
                if constexpr (boost::mp11::mp_contains<typename Coord::List, boost::mp11::mp_size_t<llama::dynamic>>::
                                  value)
                {
                    using Before = llama::mapping::internal::BeforeDynamic<Coord>;
                    using BeforeBefore = llama::RecordCoordFromList<boost::mp11::mp_pop_front<typename Before::List>>;
                    using After = llama::mapping::internal::AfterDynamic<Coord>;
                    using SubCollectionTag = llama::GetTag<AugmentedSmallEvent, Before>;

                    auto collectionColumn = ntuple->GetViewCollection(llama::structName<SubCollectionTag>());
                    auto column = collectionColumn.template GetView<Type>(
                        llama::structName<SubCollectionTag>() + "." + llama::structName<LeafTag>());
                    for (std::size_t i = 0; i < eventCount; i++)
                    {
                        const auto subCollectionCount = view(i)(BeforeBefore{})(llama::Size<SubCollectionTag>{});
                        for (std::size_t j = 0; j < subCollectionCount; j++)
                        {
                            const auto value = column(j);
                            auto& dst = view(i)(Before{})(j) (After{});
                            dst = value;
                        }
                    }
                }
                else
                {
                    auto column = ntuple->GetView<Type>(llama::structName<LeafTag>());
                    for (std::size_t i = 0; i < eventCount; i++)
                        view(i)(coord) = column(i);
                }
            }
        });
    fmt::print("Copy RNTuple -> LLAMA view: {}ms\n", duration_cast<milliseconds>(steady_clock::now() - start).count());

    start = steady_clock::now();
}
