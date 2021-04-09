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

using Event = llama::DS<
    llama::DE<tag::Muons_end, ROOT::Experimental::ClusterSize_t>
>;

using Muon = llama::DS<
    llama::DE<tag::Muon_charge, std::int32_t>,
    llama::DE<tag::Muon_phi, float>,
    llama::DE<tag::Muon_pt, float>,
    llama::DE<tag::Muon_eta, float>,
    llama::DE<tag::Muon_mass, float>
>;
// clang-format on

static void Show(TH1D& h)
{
    new TApplication("", nullptr, nullptr);

    gStyle->SetTextFont(42);
    auto c = new TCanvas("c", "", 800, 700);
    c->SetLogx();
    c->SetLogy();

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
    c->Modified();

    std::cout << "press ENTER to exit...\n";
    auto future = std::async(std::launch::async, getchar);
    while (true)
    {
        gSystem->ProcessEvents();
        if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            break;
    }
}

constexpr auto elementsPerPage = 4096;
using Page = std::vector<std::byte>;

// based on ROOT tutorial df102_NanoAODDimuonAnalysis
// download nano AOD files inside CERN:
// xrdcp "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root" \
// /tmp/Run2012B_DoubleMuParked.root
auto buildRNTupleFileModel(const std::string& path)
{
    auto copy = []<typename FieldType>(ROOT::Experimental::RNTupleView<FieldType>& view, std::vector<Page>& dstPages) {
        FieldType* dst = nullptr;
        std::size_t written = 0;
        for (auto i : view.GetFieldRange())
        {
            if (written % elementsPerPage == 0)
                dst = (FieldType*) dstPages.emplace_back(Page(sizeof(FieldType) * elementsPerPage)).data();
            dst[written % elementsPerPage] = view(i);
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
    copy(viewMuon, model["Muons_end"]);
    copy(viewCharge, model["Muon_charge"]);
    copy(viewPt, model["Muon_pt"]);
    copy(viewEta, model["Muon_eta"]);
    copy(viewPhi, model["Muon_phi"]);
    copy(viewMass, model["Muon_mass"]);

    return model;
}

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Please specify input file!\n";
        return 1;
    }

    auto rntuple = buildRNTupleFileModel(argv[1]);

    auto ts_init = std::chrono::steady_clock::now();

    auto hMass = TH1D("Dimuon_mass", "Dimuon_mass", 2000, 0.25, 300);

    std::size_t eventCount = elementsPerPage;
    std::size_t muonCount = elementsPerPage;
    auto eventView = llama::View{
        llama::mapping::SoA<llama::ArrayDomain<1>, Event, std::true_type>{llama::ArrayDomain{eventCount}},
        llama::Array<std::byte*, 1>{rntuple.at("Muons_end").front().data()}};

    auto muonView = llama::View{
        llama::mapping::SoA<llama::ArrayDomain<1>, Muon, std::true_type>{llama::ArrayDomain{muonCount}},
        llama::Array<std::byte*, 5>{
            rntuple.at("Muon_charge").front().data(),
            rntuple.at("Muon_phi").front().data(),
            rntuple.at("Muon_pt").front().data(),
            rntuple.at("Muon_eta").front().data(),
            rntuple.at("Muon_mass").front().data()}};

    const auto ts_first = std::chrono::steady_clock::now();
    for (std::size_t e = 0; e < eventCount; e++)
    {
        const auto muonOffset = e == 0 ? ROOT::Experimental::ClusterSize_t{0} : eventView(e - 1)(tag::Muons_end{});
        const auto muonCount = eventView(e)(tag::Muons_end{}) - muonOffset;
        fmt::print("Event {}, offset {}, count {}\n", e, eventView(e)(tag::Muons_end{}), muonCount);
        if (muonCount != 2)
            continue;

        if (muonOffset >= elementsPerPage) // TODO
            continue;
        // resolve to muons
        auto localMuonView = llama::VirtualView{muonView, {muonOffset}, {2}};
        if (localMuonView(0u)(tag::Muon_charge{}) == localMuonView(1u)(tag::Muon_charge{}))
            continue;

        float x_sum = 0;
        float y_sum = 0;
        float z_sum = 0;
        float e_sum = 0;
        for (std::size_t m = 0u; m < 2; ++m)
        {
            const auto x = localMuonView(m)(tag::Muon_pt{}) * std::cos(localMuonView(m)(tag::Muon_phi{}));
            x_sum += x;
            const auto y = localMuonView(m)(tag::Muon_pt{}) * std::sin(localMuonView(m)(tag::Muon_phi{}));
            y_sum += y;
            const auto z = localMuonView(m)(tag::Muon_pt{}) * std::sinh(localMuonView(m)(tag::Muon_eta{}));
            z_sum += z;
            const auto e = std::sqrt(
                x * x + y * y + z * z + localMuonView(m)(tag::Muon_mass{}) * localMuonView(m)(tag::Muon_mass{}));
            e_sum += e;
        }

        auto mass = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum - z_sum * z_sum);
        hMass.Fill(mass);
    }

    auto ts_end = std::chrono::steady_clock::now();
    auto runtime_init = std::chrono::duration_cast<std::chrono::microseconds>(ts_first - ts_init).count();
    auto runtime_analyze = std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_first).count();

    std::cout << "Runtime-Initialization: " << runtime_init << "us\n";
    std::cout << "Runtime-Analysis: " << runtime_analyze << "us\n";

    Show(hMass);

    // std::ofstream{"hep_analysis.svg"} << llama::toSvg(mapping);
    // std::ofstream{"hep_analysis.html"} << llama::toHtml(mapping);
}
