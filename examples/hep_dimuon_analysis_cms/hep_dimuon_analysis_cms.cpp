#include <RConfigure.h>
#define R__HAS_STD_STRING_VIEW
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleView.hxx>
#include <TApplication.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TSystem.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <future>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <numeric>
#include <unordered_map>

// clang-format off
namespace tag
{
    struct Muons_end{};
    struct Muon_charge{};
    struct Muon_phi{};
    struct Muon_pt{};
    struct Muon_eta{};
    struct Muon_mass{};
}

using Event = llama::Record<
    llama::Field<tag::Muons_end, ROOT::Experimental::ClusterSize_t>
>;

using Muon = llama::Record<
    llama::Field<tag::Muon_charge, std::int32_t>,
    llama::Field<tag::Muon_phi, float>,
    llama::Field<tag::Muon_pt, float>,
    llama::Field<tag::Muon_eta, float>,
    llama::Field<tag::Muon_mass, float>
>;
// clang-format on

static void Show(TH1D& h)
{
    auto app = TApplication("", nullptr, nullptr);

    gStyle->SetTextFont(42);
    auto c = TCanvas("c", "", 800, 700);
    c.SetLogx();
    c.SetLogy();

    h.SetTitle("");
    h.GetXaxis()->SetTitle("m_{#mu#mu} (GeV)");
    h.GetXaxis()->SetTitleSize(0.04);
    h.GetYaxis()->SetTitle("N_{Events}");
    h.GetYaxis()->SetTitleSize(0.04);
    h.DrawCopy();

    TLatex label;
    label.SetNDC(true);
    label.DrawLatex(0.175, 0.740, "#eta");
    label.DrawLatex(0.205, 0.775, "#rho,#omega");
    label.DrawLatex(0.270, 0.740, "#phi");
    label.DrawLatex(0.400, 0.800, "J/#psi");
    label.DrawLatex(0.415, 0.670, "#psi'");
    label.DrawLatex(0.485, 0.700, "Y(1,2,3S)");
    label.DrawLatex(0.755, 0.680, "Z");
    label.SetTextSize(0.040);
    label.DrawLatex(0.100, 0.920, "#bf{CMS Open Data}");
    label.SetTextSize(0.030);
    label.DrawLatex(0.50, 0.920, "#sqrt{s} = 8 TeV, L_{int} = 11.6 fb^{-1}");
    c.Modified();

    fmt::print("press ENTER to exit...\n");
    auto future = std::async(std::launch::async, getchar);
    while (true)
    {
        gSystem->ProcessEvents();
        if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            break;
    }
}

constexpr std::size_t elementsPerPage = 4096;
using Page = std::vector<std::byte>;

// based on ROOT tutorial df102_NanoAODDimuonAnalysis
// download nano AOD files inside CERN:
// xrdcp "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root" \
// /tmp/Run2012B_DoubleMuParked.root
auto buildRNTupleFileModel(const std::string& path)
{
    // we cannot copy the offsets stored in the RNTuple directly, because they are local to the cluster they reside in.
    // To correctly interpret this information, we would need access to the ClusterInfo stored in the RPage, which is
    // not reachable via RNTupleReader.
    auto copyOffsets = [](ROOT::Experimental::RNTupleViewCollection& view, std::vector<Page>& dstPages)
    {
        using FieldType = ROOT::Experimental::ClusterSize_t;
        FieldType* dst = nullptr;
        auto offset = FieldType{0};
        std::size_t written = 0;
        for (auto i : view.GetFieldRange())
        {
            if (written % elementsPerPage == 0)
                dst = (FieldType*) dstPages.emplace_back(Page(sizeof(FieldType) * elementsPerPage)).data();
            const auto value = view(i);
            offset += value;
            dst[written % elementsPerPage] = offset;
            // fmt::print(
            //    "i {}, offset {} stored offset {}\n",
            //    i,
            //    offset,
            //    static_cast<ROOT::Experimental::RNTupleView<FieldType>&>(view)(i));
            written++;
        }
    };

    auto copy = []<typename FieldType>(ROOT::Experimental::RNTupleView<FieldType>& view, std::vector<Page>& dstPages)
    {
        FieldType* dst = nullptr;
        std::size_t written = 0;
        for (auto i : view.GetFieldRange())
        {
            if (written % elementsPerPage == 0)
                dst = (FieldType*) dstPages.emplace_back(Page(sizeof(FieldType) * elementsPerPage)).data();
            const auto value = view(i);
            dst[written % elementsPerPage] = value;
            // fmt::print("i {} charge {}\n", written, value);
            written++;
        }
    };

    auto ntuple = ROOT::Experimental::RNTupleReader::Open(ROOT::Experimental::RNTupleModel::Create(), "NTuple", path);
    auto viewMuon = ntuple->GetViewCollection("nMuon");
    auto viewCharge = viewMuon.GetView<std::int32_t>("nMuon.Muon_charge");
    auto viewPt = viewMuon.GetView<float>("nMuon.Muon_pt");
    auto viewEta = viewMuon.GetView<float>("nMuon.Muon_eta");
    auto viewPhi = viewMuon.GetView<float>("nMuon.Muon_phi");
    auto viewMass = viewMuon.GetView<float>("nMuon.Muon_mass");

    std::unordered_map<std::string, std::vector<Page>> model;
    copyOffsets(viewMuon, model["Muons_end"]);
    copy(viewCharge, model["Muon_charge"]);
    copy(viewPt, model["Muon_pt"]);
    copy(viewEta, model["Muon_eta"]);
    copy(viewPhi, model["Muon_phi"]);
    copy(viewMass, model["Muon_mass"]);

    return std::tuple{ntuple->GetNEntries(), model};
}

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        fmt::print("Please specify input file!\n");
        return 1;
    }

    using namespace std::chrono;

    auto start = steady_clock::now();
    auto [entries, rntuple] = buildRNTupleFileModel(argv[1]);
    fmt::print("Copy RNTuple -> byte pages: {}us\n", duration_cast<microseconds>(steady_clock::now() - start).count());

    auto& Muons_endPages = rntuple.at("Muons_end");
    auto& Muon_chargePages = rntuple.at("Muon_charge");
    auto& Muon_phiPages = rntuple.at("Muon_phi");
    auto& Muon_ptPages = rntuple.at("Muon_pt");
    auto& Muon_etaPages = rntuple.at("Muon_eta");
    auto& Muon_massPages = rntuple.at("Muon_mass");

    start = std::chrono::steady_clock::now();
    auto viewEventPage = [&](std::size_t i)
    {
        return llama::View{
            llama::mapping::SoA<llama::ArrayDims<1>, Event, true>{llama::ArrayDims{elementsPerPage}},
            llama::Array<std::byte*, 1>{Muons_endPages.at(i).data()}};
    };
    auto viewMuonPage = [&](std::size_t i)
    {
        return llama::View{
            llama::mapping::SoA<llama::ArrayDims<1>, Muon, true>{llama::ArrayDims{elementsPerPage}},
            llama::Array<std::byte*, 5>{
                Muon_chargePages.at(i).data(),
                Muon_phiPages.at(i).data(),
                Muon_ptPages.at(i).data(),
                Muon_etaPages.at(i).data(),
                Muon_massPages.at(i).data()}};
    };
    fmt::print("Construct LLAMA view: {}us\n", duration_cast<microseconds>(steady_clock::now() - start).count());

    auto hMass = TH1D("Dimuon_mass", "Dimuon_mass", 2000, 0.25, 300);

    const auto pageCount = (entries + elementsPerPage - 1) / elementsPerPage;
    fmt::print("Processing {} events on {} pages\n", entries, pageCount);

    start = std::chrono::steady_clock::now();
    for (std::size_t ep = 0; ep < pageCount; ep++)
    {
        auto eventView = viewEventPage(ep);
        const auto eventsOnThisPage = std::min(elementsPerPage, entries - ep * elementsPerPage);
        for (std::size_t e = 0; e < eventsOnThisPage; e++)
        {
            const auto muonOffset = [&]()
            {
                if (e == 0)
                {
                    if (ep == 0)
                        return ROOT::Experimental::ClusterSize_t{0};
                    return viewEventPage(ep - 1)(elementsPerPage - 1)(tag::Muons_end{});
                }
                return eventView(e - 1)(tag::Muons_end{});
            }();
            const auto nextMuonOffset = eventView(e)(tag::Muons_end{});
            assert(muonOffset <= nextMuonOffset);
            const auto muonCount = nextMuonOffset - muonOffset;

            if (muonCount != 2)
                continue;

            const auto muonPageIndex = muonOffset / elementsPerPage;
            const auto muonPageInnerIndex = muonOffset % elementsPerPage;
            auto muonView = viewMuonPage(muonPageIndex);

            auto processDimuons = [&](auto dimuonView)
            {
                if (dimuonView(0u)(tag::Muon_charge{}) == dimuonView(1u)(tag::Muon_charge{}))
                    return;

                float x_sum = 0;
                float y_sum = 0;
                float z_sum = 0;
                float e_sum = 0;
                for (std::size_t m = 0u; m < 2; ++m)
                {
                    const auto x = dimuonView(m)(tag::Muon_pt{}) * std::cos(dimuonView(m)(tag::Muon_phi{}));
                    x_sum += x;
                    const auto y = dimuonView(m)(tag::Muon_pt{}) * std::sin(dimuonView(m)(tag::Muon_phi{}));
                    y_sum += y;
                    const auto z = dimuonView(m)(tag::Muon_pt{}) * std::sinh(dimuonView(m)(tag::Muon_eta{}));
                    z_sum += z;
                    const auto e = std::sqrt(
                        x * x + y * y + z * z + dimuonView(m)(tag::Muon_mass{}) * dimuonView(m)(tag::Muon_mass{}));
                    e_sum += e;
                }

                auto mass = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum - z_sum * z_sum);
                hMass.Fill(mass);
            };
            if (muonPageInnerIndex + 1 < elementsPerPage)
                processDimuons(llama::VirtualView{muonView, {muonPageInnerIndex}});
            else
            {
                constexpr auto mapping = llama::mapping::SoA<llama::ArrayDims<1>, Muon>{{2}};
                auto dimuonView = llama::allocView(mapping, llama::bloballoc::Stack<mapping.blobSize(0)>{});
                dimuonView(0u) = muonView(muonPageInnerIndex);
                dimuonView(1u) = viewMuonPage(muonPageIndex + 1)(0u);
                processDimuons(dimuonView);
            }
        }
    }

    fmt::print("Analysis: {}us\n", duration_cast<microseconds>(steady_clock::now() - start).count());

    Show(hMass);
}
