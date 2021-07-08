#pragma once

// This record dimension example is inspired by a non-public CMS NanoAOD file called: ttjet_13tev_june2019_lzma.

#include <cstdint>
#include <llama/llama.hpp>

using bit = bool;
using byte = unsigned char;
using Index = std::uint32_t;

// clang-format off
struct run {};
struct luminosityBlock {};
struct event {};
struct HTXS_Higgs_pt {};
struct HTXS_Higgs_y {};
struct HTXS_stage1_1_cat_pTjet25GeV {};
struct HTXS_stage1_1_cat_pTjet30GeV {};
struct HTXS_stage1_1_fine_cat_pTjet25GeV {};
struct HTXS_stage1_1_fine_cat_pTjet30GeV {};
struct HTXS_stage_0 {};
struct HTXS_stage_1_pTjet25 {};
struct HTXS_stage_1_pTjet30 {};
struct HTXS_njets25 {};
struct HTXS_njets30 {};
struct btagWeight_CSVV2 {};
struct btagWeight_DeepCSVB {};
struct CaloMET_phi {};
struct CaloMET_pt {};
struct CaloMET_sumEt {};
struct ChsMET_phi {};
struct ChsMET_pt {};
struct ChsMET_sumEt {};
struct nCorrT1METJet {};
struct CorrT1METJet_area {};
struct CorrT1METJet_eta {};
struct CorrT1METJet_muonSubtrFactor {};
struct CorrT1METJet_phi {};
struct CorrT1METJet_rawPt {};
struct nElectron {};
struct Electron_deltaEtaSC {};
struct Electron_dr03EcalRecHitSumEt {};
struct Electron_dr03HcalDepth1TowerSumEt {};
struct Electron_dr03TkSumPt {};
struct Electron_dr03TkSumPtHEEP {};
struct Electron_dxy {};
struct Electron_dxyErr {};
struct Electron_dz {};
struct Electron_dzErr {};
struct Electron_eCorr {};
struct Electron_eInvMinusPInv {};
struct Electron_energyErr {};
struct Electron_eta {};
struct Electron_hoe {};
struct Electron_ip3d {};
struct Electron_jetPtRelv2 {};
struct Electron_jetRelIso {};
struct Electron_mass {};
struct Electron_miniPFRelIso_all {};
struct Electron_miniPFRelIso_chg {};
struct Electron_mvaFall17V1Iso {};
struct Electron_mvaFall17V1noIso {};
struct Electron_mvaFall17V2Iso {};
struct Electron_mvaFall17V2noIso {};
struct Electron_pfRelIso03_all {};
struct Electron_pfRelIso03_chg {};
struct Electron_phi {};
struct Electron_pt {};
struct Electron_r9 {};
struct Electron_sieie {};
struct Electron_sip3d {};
struct Electron_mvaTTH {};
struct Electron_charge {};
struct Electron_cutBased {};
struct Electron_cutBased_Fall17_V1 {};
struct Electron_jetIdx {};
struct Electron_pdgId {};
struct Electron_photonIdx {};
struct Electron_tightCharge {};
struct Electron_vidNestedWPbitmap {};
struct Electron_convVeto {};
struct Electron_cutBased_HEEP {};
struct Electron_isPFcand {};
struct Electron_lostHits {};
struct Electron_mvaFall17V1Iso_WP80 {};
struct Electron_mvaFall17V1Iso_WP90 {};
struct Electron_mvaFall17V1Iso_WPL {};
struct Electron_mvaFall17V1noIso_WP80 {};
struct Electron_mvaFall17V1noIso_WP90 {};
struct Electron_mvaFall17V1noIso_WPL {};
struct Electron_mvaFall17V2Iso_WP80 {};
struct Electron_mvaFall17V2Iso_WP90 {};
struct Electron_mvaFall17V2Iso_WPL {};
struct Electron_mvaFall17V2noIso_WP80 {};
struct Electron_mvaFall17V2noIso_WP90 {};
struct Electron_mvaFall17V2noIso_WPL {};
struct Electron_seedGain {};
struct Electron_genPartIdx {};
struct Electron_genPartFlav {};
struct Electron_cleanmask {};
struct Flag_ecalBadCalibFilterV2 {};
struct nFatJet {};
struct FatJet_area {};
struct FatJet_btagCMVA {};
struct FatJet_btagCSVV2 {};
struct FatJet_btagDDBvL {};
struct FatJet_btagDDCvB {};
struct FatJet_btagDDCvL {};
struct FatJet_btagDeepB {};
struct FatJet_btagHbb {};
struct FatJet_deepTagMD_H4qvsQCD {};
struct FatJet_deepTagMD_HbbvsQCD {};
struct FatJet_deepTagMD_TvsQCD {};
struct FatJet_deepTagMD_WvsQCD {};
struct FatJet_deepTagMD_ZHbbvsQCD {};
struct FatJet_deepTagMD_ZHccvsQCD {};
struct FatJet_deepTagMD_ZbbvsQCD {};
struct FatJet_deepTagMD_ZvsQCD {};
struct FatJet_deepTagMD_bbvsLight {};
struct FatJet_deepTagMD_ccvsLight {};
struct FatJet_deepTag_H {};
struct FatJet_deepTag_QCD {};
struct FatJet_deepTag_QCDothers {};
struct FatJet_deepTag_TvsQCD {};
struct FatJet_deepTag_WvsQCD {};
struct FatJet_deepTag_ZvsQCD {};
struct FatJet_eta {};
struct FatJet_mass {};
struct FatJet_msoftdrop {};
struct FatJet_n2b1 {};
struct FatJet_n3b1 {};
struct FatJet_phi {};
struct FatJet_pt {};
struct FatJet_rawFactor {};
struct FatJet_tau1 {};
struct FatJet_tau2 {};
struct FatJet_tau3 {};
struct FatJet_tau4 {};
struct FatJet_jetId {};
struct FatJet_subJetIdx1 {};
struct FatJet_subJetIdx2 {};
struct nGenJetAK8 {};
struct GenJetAK8_eta {};
struct GenJetAK8_mass {};
struct GenJetAK8_phi {};
struct GenJetAK8_pt {};
struct GenJetAK8_partonFlavour {};
struct GenJetAK8_hadronFlavour {};
struct nGenJet {};
struct GenJet_eta {};
struct GenJet_mass {};
struct GenJet_phi {};
struct GenJet_pt {};
struct GenJet_partonFlavour {};
struct GenJet_hadronFlavour {};
struct nGenPart {};
struct GenPart_eta {};
struct GenPart_mass {};
struct GenPart_phi {};
struct GenPart_pt {};
struct GenPart_genPartIdxMother {};
struct GenPart_pdgId {};
struct GenPart_status {};
struct GenPart_statusFlags {};
struct nSubGenJetAK8 {};
struct SubGenJetAK8_eta {};
struct SubGenJetAK8_mass {};
struct SubGenJetAK8_phi {};
struct SubGenJetAK8_pt {};
struct Generator_binvar {};
struct Generator_scalePDF {};
struct Generator_weight {};
struct Generator_x1 {};
struct Generator_x2 {};
struct Generator_xpdf1 {};
struct Generator_xpdf2 {};
struct Generator_id1 {};
struct Generator_id2 {};
struct nGenVisTau {};
struct GenVisTau_eta {};
struct GenVisTau_mass {};
struct GenVisTau_phi {};
struct GenVisTau_pt {};
struct GenVisTau_charge {};
struct GenVisTau_genPartIdxMother {};
struct GenVisTau_status {};
struct genWeight {};
struct LHEWeight_originalXWGTUP {};
struct nLHEPdfWeight {};
struct LHEPdfWeight {};
struct nLHEReweightingWeight {};
struct LHEReweightingWeight {};
struct nLHEScaleWeight {};
struct LHEScaleWeight {};
struct nPSWeight {};
struct PSWeight {};
struct nIsoTrack {};
struct IsoTrack_dxy {};
struct IsoTrack_dz {};
struct IsoTrack_eta {};
struct IsoTrack_pfRelIso03_all {};
struct IsoTrack_pfRelIso03_chg {};
struct IsoTrack_phi {};
struct IsoTrack_pt {};
struct IsoTrack_miniPFRelIso_all {};
struct IsoTrack_miniPFRelIso_chg {};
struct IsoTrack_fromPV {};
struct IsoTrack_pdgId {};
struct IsoTrack_isHighPurityTrack {};
struct IsoTrack_isPFcand {};
struct IsoTrack_isFromLostTrack {};
struct nJet {};
struct Jet_area {};
struct Jet_btagCMVA {};
struct Jet_btagCSVV2 {};
struct Jet_btagDeepB {};
struct Jet_btagDeepC {};
struct Jet_btagDeepFlavB {};
struct Jet_btagDeepFlavC {};
struct Jet_chEmEF {};
struct Jet_chHEF {};
struct Jet_eta {};
struct Jet_jercCHF {};
struct Jet_jercCHPUF {};
struct Jet_mass {};
struct Jet_muEF {};
struct Jet_muonSubtrFactor {};
struct Jet_neEmEF {};
struct Jet_neHEF {};
struct Jet_phi {};
struct Jet_pt {};
struct Jet_qgl {};
struct Jet_rawFactor {};
struct Jet_bRegCorr {};
struct Jet_bRegRes {};
struct Jet_electronIdx1 {};
struct Jet_electronIdx2 {};
struct Jet_jetId {};
struct Jet_muonIdx1 {};
struct Jet_muonIdx2 {};
struct Jet_nConstituents {};
struct Jet_nElectrons {};
struct Jet_nMuons {};
struct Jet_puId {};
struct Jet_genJetIdx {};
struct Jet_hadronFlavour {};
struct Jet_partonFlavour {};
struct Jet_cleanmask {};
struct LHE_HT {};
struct LHE_HTIncoming {};
struct LHE_Vpt {};
struct LHE_Njets {};
struct LHE_Nb {};
struct LHE_Nc {};
struct LHE_Nuds {};
struct LHE_Nglu {};
struct LHE_NpNLO {};
struct LHE_NpLO {};
struct nLHEPart {};
struct LHEPart_pt {};
struct LHEPart_eta {};
struct LHEPart_phi {};
struct LHEPart_mass {};
struct LHEPart_pdgId {};
struct GenMET_phi {};
struct GenMET_pt {};
struct MET_MetUnclustEnUpDeltaX {};
struct MET_MetUnclustEnUpDeltaY {};
struct MET_covXX {};
struct MET_covXY {};
struct MET_covYY {};
struct MET_phi {};
struct MET_pt {};
struct MET_significance {};
struct MET_sumEt {};
struct nMuon {};
struct Muon_dxy {};
struct Muon_dxyErr {};
struct Muon_dz {};
struct Muon_dzErr {};
struct Muon_eta {};
struct Muon_ip3d {};
struct Muon_jetPtRelv2 {};
struct Muon_jetRelIso {};
struct Muon_mass {};
struct Muon_miniPFRelIso_all {};
struct Muon_miniPFRelIso_chg {};
struct Muon_pfRelIso03_all {};
struct Muon_pfRelIso03_chg {};
struct Muon_pfRelIso04_all {};
struct Muon_phi {};
struct Muon_pt {};
struct Muon_ptErr {};
struct Muon_segmentComp {};
struct Muon_sip3d {};
struct Muon_softMva {};
struct Muon_tkRelIso {};
struct Muon_tunepRelPt {};
struct Muon_mvaLowPt {};
struct Muon_mvaTTH {};
struct Muon_charge {};
struct Muon_jetIdx {};
struct Muon_nStations {};
struct Muon_nTrackerLayers {};
struct Muon_pdgId {};
struct Muon_tightCharge {};
struct Muon_highPtId {};
struct Muon_inTimeMuon {};
struct Muon_isGlobal {};
struct Muon_isPFcand {};
struct Muon_isTracker {};
struct Muon_looseId {};
struct Muon_mediumId {};
struct Muon_mediumPromptId {};
struct Muon_miniIsoId {};
struct Muon_multiIsoId {};
struct Muon_mvaId {};
struct Muon_pfIsoId {};
struct Muon_softId {};
struct Muon_softMvaId {};
struct Muon_tightId {};
struct Muon_tkIsoId {};
struct Muon_triggerIdLoose {};
struct Muon_genPartIdx {};
struct Muon_genPartFlav {};
struct Muon_cleanmask {};
struct nPhoton {};
struct Photon_eCorr {};
struct Photon_energyErr {};
struct Photon_eta {};
struct Photon_hoe {};
struct Photon_mass {};
struct Photon_mvaID {};
struct Photon_mvaIDV1 {};
struct Photon_pfRelIso03_all {};
struct Photon_pfRelIso03_chg {};
struct Photon_phi {};
struct Photon_pt {};
struct Photon_r9 {};
struct Photon_sieie {};
struct Photon_charge {};
struct Photon_cutBasedbitmap {};
struct Photon_cutBasedV1bitmap {};
struct Photon_electronIdx {};
struct Photon_jetIdx {};
struct Photon_pdgId {};
struct Photon_vidNestedWPbitmap {};
struct Photon_electronVeto {};
struct Photon_isScEtaEB {};
struct Photon_isScEtaEE {};
struct Photon_mvaID_WP80 {};
struct Photon_mvaID_WP90 {};
struct Photon_pixelSeed {};
struct Photon_seedGain {};
struct Photon_genPartIdx {};
struct Photon_genPartFlav {};
struct Photon_cleanmask {};
struct Pileup_nTrueInt {};
struct Pileup_pudensity {};
struct Pileup_gpudensity {};
struct Pileup_nPU {};
struct Pileup_sumEOOT {};
struct Pileup_sumLOOT {};
struct PuppiMET_phi {};
struct PuppiMET_pt {};
struct PuppiMET_sumEt {};
struct RawMET_phi {};
struct RawMET_pt {};
struct RawMET_sumEt {};
struct fixedGridRhoFastjetAll {};
struct fixedGridRhoFastjetCentral {};
struct fixedGridRhoFastjetCentralCalo {};
struct fixedGridRhoFastjetCentralChargedPileUp {};
struct fixedGridRhoFastjetCentralNeutral {};
struct nGenDressedLepton {};
struct GenDressedLepton_eta {};
struct GenDressedLepton_mass {};
struct GenDressedLepton_phi {};
struct GenDressedLepton_pt {};
struct GenDressedLepton_pdgId {};
struct GenDressedLepton_hasTauAnc {};
struct nSoftActivityJet {};
struct SoftActivityJet_eta {};
struct SoftActivityJet_phi {};
struct SoftActivityJet_pt {};
struct SoftActivityJetHT {};
struct SoftActivityJetHT10 {};
struct SoftActivityJetHT2 {};
struct SoftActivityJetHT5 {};
struct SoftActivityJetNjets10 {};
struct SoftActivityJetNjets2 {};
struct SoftActivityJetNjets5 {};
struct nSubJet {};
struct SubJet_btagCMVA {};
struct SubJet_btagCSVV2 {};
struct SubJet_btagDeepB {};
struct SubJet_eta {};
struct SubJet_mass {};
struct SubJet_n2b1 {};
struct SubJet_n3b1 {};
struct SubJet_phi {};
struct SubJet_pt {};
struct SubJet_rawFactor {};
struct SubJet_tau1 {};
struct SubJet_tau2 {};
struct SubJet_tau3 {};
struct SubJet_tau4 {};
struct nTau {};
struct Tau_chargedIso {};
struct Tau_dxy {};
struct Tau_dz {};
struct Tau_eta {};
struct Tau_leadTkDeltaEta {};
struct Tau_leadTkDeltaPhi {};
struct Tau_leadTkPtOverTauPt {};
struct Tau_mass {};
struct Tau_neutralIso {};
struct Tau_phi {};
struct Tau_photonsOutsideSignalCone {};
struct Tau_pt {};
struct Tau_puCorr {};
struct Tau_rawAntiEle {};
struct Tau_rawAntiEle2018 {};
struct Tau_rawIso {};
struct Tau_rawIsodR03 {};
struct Tau_rawMVAnewDM2017v2 {};
struct Tau_rawMVAoldDM {};
struct Tau_rawMVAoldDM2017v1 {};
struct Tau_rawMVAoldDM2017v2 {};
struct Tau_rawMVAoldDMdR032017v2 {};
struct Tau_charge {};
struct Tau_decayMode {};
struct Tau_jetIdx {};
struct Tau_rawAntiEleCat {};
struct Tau_rawAntiEleCat2018 {};
struct Tau_idAntiEle {};
struct Tau_idAntiEle2018 {};
struct Tau_idAntiMu {};
struct Tau_idDecayMode {};
struct Tau_idDecayModeNewDMs {};
struct Tau_idMVAnewDM2017v2 {};
struct Tau_idMVAoldDM {};
struct Tau_idMVAoldDM2017v1 {};
struct Tau_idMVAoldDM2017v2 {};
struct Tau_idMVAoldDMdR032017v2 {};
struct Tau_cleanmask {};
struct Tau_genPartIdx {};
struct Tau_genPartFlav {};
struct TkMET_phi {};
struct TkMET_pt {};
struct TkMET_sumEt {};
struct nTrigObj {};
struct TrigObj_pt {};
struct TrigObj_eta {};
struct TrigObj_phi {};
struct TrigObj_l1pt {};
struct TrigObj_l1pt_2 {};
struct TrigObj_l2pt {};
struct TrigObj_id {};
struct TrigObj_l1iso {};
struct TrigObj_l1charge {};
struct TrigObj_filterbits {};
struct genTtbarId {};
struct nOtherPV {};
struct OtherPV_z {};
struct PV_ndof {};
struct PV_x {};
struct PV_y {};
struct PV_z {};
struct PV_chi2 {};
struct PV_score {};
struct PV_npvs {};
struct PV_npvsGood {};
struct nSV {};
struct SV_dlen {};
struct SV_dlenSig {};
struct SV_pAngle {};
struct SV_chi2 {};
struct SV_eta {};
struct SV_mass {};
struct SV_ndof {};
struct SV_phi {};
struct SV_pt {};
struct SV_x {};
struct SV_y {};
struct SV_z {};
struct MET_fiducialGenPhi {};
struct MET_fiducialGenPt {};
struct L1simulation_step {};
struct HLTriggerFirstPath {};
struct HLT_AK8PFJet360_TrimMass30 {};
struct HLT_AK8PFJet380_TrimMass30 {};
struct HLT_AK8PFJet400_TrimMass30 {};
struct HLT_AK8PFJet420_TrimMass30 {};
struct HLT_AK8PFHT750_TrimMass50 {};
struct HLT_AK8PFHT800_TrimMass50 {};
struct HLT_AK8PFHT850_TrimMass50 {};
struct HLT_AK8PFHT900_TrimMass50 {};
struct HLT_CaloJet500_NoJetID {};
struct HLT_CaloJet550_NoJetID {};
struct HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL {};
struct HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon {};
struct HLT_Trimuon5_3p5_2_Upsilon_Muon {};
struct HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon {};
struct HLT_DoubleEle25_CaloIdL_MW {};
struct HLT_DoubleEle27_CaloIdL_MW {};
struct HLT_DoubleEle33_CaloIdL_MW {};
struct HLT_DoubleEle24_eta2p1_WPTight_Gsf {};
struct HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350 {};
struct HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350 {};
struct HLT_Ele27_Ele37_CaloIdL_MW {};
struct HLT_Mu27_Ele37_CaloIdL_MW {};
struct HLT_Mu37_Ele27_CaloIdL_MW {};
struct HLT_Mu37_TkMu27 {};
struct HLT_DoubleMu4_3_Bs {};
struct HLT_DoubleMu4_3_Jpsi {};
struct HLT_DoubleMu4_JpsiTrk_Displaced {};
struct HLT_DoubleMu4_LowMassNonResonantTrk_Displaced {};
struct HLT_DoubleMu3_Trk_Tau3mu {};
struct HLT_DoubleMu3_TkMu_DsTau3Mu {};
struct HLT_DoubleMu4_PsiPrimeTrk_Displaced {};
struct HLT_DoubleMu4_Mass3p8_DZ_PFHT350 {};
struct HLT_Mu3_PFJet40 {};
struct HLT_Mu7p5_L2Mu2_Jpsi {};
struct HLT_Mu7p5_L2Mu2_Upsilon {};
struct HLT_Mu7p5_Track2_Jpsi {};
struct HLT_Mu7p5_Track3p5_Jpsi {};
struct HLT_Mu7p5_Track7_Jpsi {};
struct HLT_Mu7p5_Track2_Upsilon {};
struct HLT_Mu7p5_Track3p5_Upsilon {};
struct HLT_Mu7p5_Track7_Upsilon {};
struct HLT_Mu3_L1SingleMu5orSingleMu7 {};
struct HLT_DoublePhoton33_CaloIdL {};
struct HLT_DoublePhoton70 {};
struct HLT_DoublePhoton85 {};
struct HLT_Ele20_WPTight_Gsf {};
struct HLT_Ele15_WPLoose_Gsf {};
struct HLT_Ele17_WPLoose_Gsf {};
struct HLT_Ele20_WPLoose_Gsf {};
struct HLT_Ele20_eta2p1_WPLoose_Gsf {};
struct HLT_DiEle27_WPTightCaloOnly_L1DoubleEG {};
struct HLT_Ele27_WPTight_Gsf {};
struct HLT_Ele28_WPTight_Gsf {};
struct HLT_Ele30_WPTight_Gsf {};
struct HLT_Ele32_WPTight_Gsf {};
struct HLT_Ele35_WPTight_Gsf {};
struct HLT_Ele35_WPTight_Gsf_L1EGMT {};
struct HLT_Ele38_WPTight_Gsf {};
struct HLT_Ele40_WPTight_Gsf {};
struct HLT_Ele32_WPTight_Gsf_L1DoubleEG {};
struct HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1 {};
struct HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1 {};
struct HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1 {};
struct HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1 {};
struct HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1 {};
struct HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1 {};
struct HLT_HT450_Beamspot {};
struct HLT_HT300_Beamspot {};
struct HLT_ZeroBias_Beamspot {};
struct HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1 {};
struct HLT_IsoMu20_eta2p1_MediumChargedIsoPFTauHPS27_eta2p1_CrossL1 {};
struct HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_CrossL1 {};
struct HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1 {};
struct HLT_IsoMu20_eta2p1_MediumChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1 {};
struct HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1 {};
struct HLT_IsoMu24_eta2p1_TightChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_CrossL1 {};
struct HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_CrossL1 {};
struct HLT_IsoMu24_eta2p1_TightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_CrossL1 {};
struct HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_CrossL1 {};
struct HLT_IsoMu27_LooseChargedIsoPFTauHPS20_Trk1_eta2p1_SingleL1 {};
struct HLT_IsoMu27_MediumChargedIsoPFTauHPS20_Trk1_eta2p1_SingleL1 {};
struct HLT_IsoMu27_TightChargedIsoPFTauHPS20_Trk1_eta2p1_SingleL1 {};
struct HLT_IsoMu20 {};
struct HLT_IsoMu24 {};
struct HLT_IsoMu24_eta2p1 {};
struct HLT_IsoMu27 {};
struct HLT_IsoMu30 {};
struct HLT_UncorrectedJetE30_NoBPTX {};
struct HLT_UncorrectedJetE30_NoBPTX3BX {};
struct HLT_UncorrectedJetE60_NoBPTX3BX {};
struct HLT_UncorrectedJetE70_NoBPTX3BX {};
struct HLT_L1SingleMu18 {};
struct HLT_L1SingleMu25 {};
struct HLT_L2Mu10 {};
struct HLT_L2Mu10_NoVertex_NoBPTX3BX {};
struct HLT_L2Mu10_NoVertex_NoBPTX {};
struct HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX {};
struct HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX {};
struct HLT_L2Mu50 {};
struct HLT_L2Mu23NoVtx_2Cha {};
struct HLT_L2Mu23NoVtx_2Cha_CosmicSeed {};
struct HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4 {};
struct HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4 {};
struct HLT_DoubleL2Mu50 {};
struct HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed {};
struct HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_NoL2Matched {};
struct HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed {};
struct HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_NoL2Matched {};
struct HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4 {};
struct HLT_DoubleL2Mu23NoVtx_2Cha {};
struct HLT_DoubleL2Mu23NoVtx_2Cha_NoL2Matched {};
struct HLT_DoubleL2Mu25NoVtx_2Cha {};
struct HLT_DoubleL2Mu25NoVtx_2Cha_NoL2Matched {};
struct HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4 {};
struct HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL {};
struct HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL {};
struct HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ {};
struct HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ {};
struct HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 {};
struct HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8 {};
struct HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 {};
struct HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8 {};
struct HLT_Mu25_TkMu0_Onia {};
struct HLT_Mu30_TkMu0_Psi {};
struct HLT_Mu30_TkMu0_Upsilon {};
struct HLT_Mu20_TkMu0_Phi {};
struct HLT_Mu25_TkMu0_Phi {};
struct HLT_Mu12 {};
struct HLT_Mu15 {};
struct HLT_Mu20 {};
struct HLT_Mu27 {};
struct HLT_Mu50 {};
struct HLT_Mu55 {};
struct HLT_OldMu100 {};
struct HLT_TkMu100 {};
struct HLT_DiPFJetAve40 {};
struct HLT_DiPFJetAve60 {};
struct HLT_DiPFJetAve80 {};
struct HLT_DiPFJetAve140 {};
struct HLT_DiPFJetAve200 {};
struct HLT_DiPFJetAve260 {};
struct HLT_DiPFJetAve320 {};
struct HLT_DiPFJetAve400 {};
struct HLT_DiPFJetAve500 {};
struct HLT_DiPFJetAve60_HFJEC {};
struct HLT_DiPFJetAve80_HFJEC {};
struct HLT_DiPFJetAve100_HFJEC {};
struct HLT_DiPFJetAve160_HFJEC {};
struct HLT_DiPFJetAve220_HFJEC {};
struct HLT_DiPFJetAve300_HFJEC {};
struct HLT_AK8PFJet15 {};
struct HLT_AK8PFJet25 {};
struct HLT_AK8PFJet40 {};
struct HLT_AK8PFJet60 {};
struct HLT_AK8PFJet80 {};
struct HLT_AK8PFJet140 {};
struct HLT_AK8PFJet200 {};
struct HLT_AK8PFJet260 {};
struct HLT_AK8PFJet320 {};
struct HLT_AK8PFJet400 {};
struct HLT_AK8PFJet450 {};
struct HLT_AK8PFJet500 {};
struct HLT_AK8PFJet550 {};
struct HLT_PFJet15 {};
struct HLT_PFJet25 {};
struct HLT_PFJet40 {};
struct HLT_PFJet60 {};
struct HLT_PFJet80 {};
struct HLT_PFJet140 {};
struct HLT_PFJet200 {};
struct HLT_PFJet260 {};
struct HLT_PFJet320 {};
struct HLT_PFJet400 {};
struct HLT_PFJet450 {};
struct HLT_PFJet500 {};
struct HLT_PFJet550 {};
struct HLT_PFJetFwd15 {};
struct HLT_PFJetFwd25 {};
struct HLT_PFJetFwd40 {};
struct HLT_PFJetFwd60 {};
struct HLT_PFJetFwd80 {};
struct HLT_PFJetFwd140 {};
struct HLT_PFJetFwd200 {};
struct HLT_PFJetFwd260 {};
struct HLT_PFJetFwd320 {};
struct HLT_PFJetFwd400 {};
struct HLT_PFJetFwd450 {};
struct HLT_PFJetFwd500 {};
struct HLT_AK8PFJetFwd15 {};
struct HLT_AK8PFJetFwd25 {};
struct HLT_AK8PFJetFwd40 {};
struct HLT_AK8PFJetFwd60 {};
struct HLT_AK8PFJetFwd80 {};
struct HLT_AK8PFJetFwd140 {};
struct HLT_AK8PFJetFwd200 {};
struct HLT_AK8PFJetFwd260 {};
struct HLT_AK8PFJetFwd320 {};
struct HLT_AK8PFJetFwd400 {};
struct HLT_AK8PFJetFwd450 {};
struct HLT_AK8PFJetFwd500 {};
struct HLT_PFHT180 {};
struct HLT_PFHT250 {};
struct HLT_PFHT370 {};
struct HLT_PFHT430 {};
struct HLT_PFHT510 {};
struct HLT_PFHT590 {};
struct HLT_PFHT680 {};
struct HLT_PFHT780 {};
struct HLT_PFHT890 {};
struct HLT_PFHT1050 {};
struct HLT_PFHT500_PFMET100_PFMHT100_IDTight {};
struct HLT_PFHT500_PFMET110_PFMHT110_IDTight {};
struct HLT_PFHT700_PFMET85_PFMHT85_IDTight {};
struct HLT_PFHT700_PFMET95_PFMHT95_IDTight {};
struct HLT_PFHT800_PFMET75_PFMHT75_IDTight {};
struct HLT_PFHT800_PFMET85_PFMHT85_IDTight {};
struct HLT_PFMET110_PFMHT110_IDTight {};
struct HLT_PFMET120_PFMHT120_IDTight {};
struct HLT_PFMET130_PFMHT130_IDTight {};
struct HLT_PFMET140_PFMHT140_IDTight {};
struct HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1 {};
struct HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1 {};
struct HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1 {};
struct HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1 {};
struct HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1 {};
struct HLT_PFMET120_PFMHT120_IDTight_PFHT60 {};
struct HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 {};
struct HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60 {};
struct HLT_PFMETTypeOne110_PFMHT110_IDTight {};
struct HLT_PFMETTypeOne120_PFMHT120_IDTight {};
struct HLT_PFMETTypeOne130_PFMHT130_IDTight {};
struct HLT_PFMETTypeOne140_PFMHT140_IDTight {};
struct HLT_PFMETNoMu110_PFMHTNoMu110_IDTight {};
struct HLT_PFMETNoMu120_PFMHTNoMu120_IDTight {};
struct HLT_PFMETNoMu130_PFMHTNoMu130_IDTight {};
struct HLT_PFMETNoMu140_PFMHTNoMu140_IDTight {};
struct HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight {};
struct HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight {};
struct HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight {};
struct HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight {};
struct HLT_L1ETMHadSeeds {};
struct HLT_CaloMHT90 {};
struct HLT_CaloMET80_NotCleaned {};
struct HLT_CaloMET90_NotCleaned {};
struct HLT_CaloMET100_NotCleaned {};
struct HLT_CaloMET110_NotCleaned {};
struct HLT_CaloMET250_NotCleaned {};
struct HLT_CaloMET70_HBHECleaned {};
struct HLT_CaloMET80_HBHECleaned {};
struct HLT_CaloMET90_HBHECleaned {};
struct HLT_CaloMET100_HBHECleaned {};
struct HLT_CaloMET250_HBHECleaned {};
struct HLT_CaloMET300_HBHECleaned {};
struct HLT_CaloMET350_HBHECleaned {};
struct HLT_PFMET200_NotCleaned {};
struct HLT_PFMET200_HBHECleaned {};
struct HLT_PFMET250_HBHECleaned {};
struct HLT_PFMET300_HBHECleaned {};
struct HLT_PFMET200_HBHE_BeamHaloCleaned {};
struct HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned {};
struct HLT_MET105_IsoTrk50 {};
struct HLT_MET120_IsoTrk50 {};
struct HLT_SingleJet30_Mu12_SinglePFJet40 {};
struct HLT_Mu12_DoublePFJets40_CaloBTagDeepCSV_p71 {};
struct HLT_Mu12_DoublePFJets100_CaloBTagDeepCSV_p71 {};
struct HLT_Mu12_DoublePFJets200_CaloBTagDeepCSV_p71 {};
struct HLT_Mu12_DoublePFJets350_CaloBTagDeepCSV_p71 {};
struct HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 {};
struct HLT_Mu12_DoublePFJets54MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 {};
struct HLT_Mu12_DoublePFJets62MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 {};
struct HLT_DoublePFJets40_CaloBTagDeepCSV_p71 {};
struct HLT_DoublePFJets100_CaloBTagDeepCSV_p71 {};
struct HLT_DoublePFJets200_CaloBTagDeepCSV_p71 {};
struct HLT_DoublePFJets350_CaloBTagDeepCSV_p71 {};
struct HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 {};
struct HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 {};
struct HLT_Photon300_NoHE {};
struct HLT_Mu8_TrkIsoVVL {};
struct HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ {};
struct HLT_Mu8_DiEle12_CaloIdL_TrackIdL {};
struct HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ {};
struct HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350 {};
struct HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ {};
struct HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30 {};
struct HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30 {};
struct HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5 {};
struct HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5 {};
struct HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL {};
struct HLT_Mu17_TrkIsoVVL {};
struct HLT_Mu19_TrkIsoVVL {};
struct HLT_BTagMu_AK4DiJet20_Mu5 {};
struct HLT_BTagMu_AK4DiJet40_Mu5 {};
struct HLT_BTagMu_AK4DiJet70_Mu5 {};
struct HLT_BTagMu_AK4DiJet110_Mu5 {};
struct HLT_BTagMu_AK4DiJet170_Mu5 {};
struct HLT_BTagMu_AK4Jet300_Mu5 {};
struct HLT_BTagMu_AK8DiJet170_Mu5 {};
struct HLT_BTagMu_AK8Jet170_DoubleMu5 {};
struct HLT_BTagMu_AK8Jet300_Mu5 {};
struct HLT_BTagMu_AK4DiJet20_Mu5_noalgo {};
struct HLT_BTagMu_AK4DiJet40_Mu5_noalgo {};
struct HLT_BTagMu_AK4DiJet70_Mu5_noalgo {};
struct HLT_BTagMu_AK4DiJet110_Mu5_noalgo {};
struct HLT_BTagMu_AK4DiJet170_Mu5_noalgo {};
struct HLT_BTagMu_AK4Jet300_Mu5_noalgo {};
struct HLT_BTagMu_AK8DiJet170_Mu5_noalgo {};
struct HLT_BTagMu_AK8Jet170_DoubleMu5_noalgo {};
struct HLT_BTagMu_AK8Jet300_Mu5_noalgo {};
struct HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL {};
struct HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ {};
struct HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL {};
struct HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ {};
struct HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL {};
struct HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL {};
struct HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ {};
struct HLT_Mu12_DoublePhoton20 {};
struct HLT_TriplePhoton_20_20_20_CaloIdLV2 {};
struct HLT_TriplePhoton_20_20_20_CaloIdLV2_R9IdVL {};
struct HLT_TriplePhoton_30_30_10_CaloIdLV2 {};
struct HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL {};
struct HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL {};
struct HLT_Photon20 {};
struct HLT_Photon33 {};
struct HLT_Photon50 {};
struct HLT_Photon75 {};
struct HLT_Photon90 {};
struct HLT_Photon120 {};
struct HLT_Photon150 {};
struct HLT_Photon175 {};
struct HLT_Photon200 {};
struct HLT_Photon100EB_TightID_TightIso {};
struct HLT_Photon110EB_TightID_TightIso {};
struct HLT_Photon120EB_TightID_TightIso {};
struct HLT_Photon100EBHE10 {};
struct HLT_Photon100EEHE10 {};
struct HLT_Photon100EE_TightID_TightIso {};
struct HLT_Photon50_R9Id90_HE10_IsoM {};
struct HLT_Photon75_R9Id90_HE10_IsoM {};
struct HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3 {};
struct HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3 {};
struct HLT_Photon90_R9Id90_HE10_IsoM {};
struct HLT_Photon120_R9Id90_HE10_IsoM {};
struct HLT_Photon165_R9Id90_HE10_IsoM {};
struct HLT_Photon90_CaloIdL_PFHT700 {};
struct HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90 {};
struct HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95 {};
struct HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55 {};
struct HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55 {};
struct HLT_Photon35_TwoProngs35 {};
struct HLT_IsoMu24_TwoProngs35 {};
struct HLT_Dimuon0_Jpsi_L1_NoOS {};
struct HLT_Dimuon0_Jpsi_NoVertexing_NoOS {};
struct HLT_Dimuon0_Jpsi {};
struct HLT_Dimuon0_Jpsi_NoVertexing {};
struct HLT_Dimuon0_Jpsi_L1_4R_0er1p5R {};
struct HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R {};
struct HLT_Dimuon0_Jpsi3p5_Muon2 {};
struct HLT_Dimuon0_Upsilon_L1_4p5 {};
struct HLT_Dimuon0_Upsilon_L1_5 {};
struct HLT_Dimuon0_Upsilon_L1_4p5NoOS {};
struct HLT_Dimuon0_Upsilon_L1_4p5er2p0 {};
struct HLT_Dimuon0_Upsilon_L1_4p5er2p0M {};
struct HLT_Dimuon0_Upsilon_NoVertexing {};
struct HLT_Dimuon0_Upsilon_L1_5M {};
struct HLT_Dimuon0_LowMass_L1_0er1p5R {};
struct HLT_Dimuon0_LowMass_L1_0er1p5 {};
struct HLT_Dimuon0_LowMass {};
struct HLT_Dimuon0_LowMass_L1_4 {};
struct HLT_Dimuon0_LowMass_L1_4R {};
struct HLT_Dimuon0_LowMass_L1_TM530 {};
struct HLT_Dimuon0_Upsilon_Muon_L1_TM0 {};
struct HLT_Dimuon0_Upsilon_Muon_NoL1Mass {};
struct HLT_TripleMu_5_3_3_Mass3p8_DZ {};
struct HLT_TripleMu_10_5_5_DZ {};
struct HLT_TripleMu_12_10_5 {};
struct HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15 {};
struct HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1 {};
struct HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15 {};
struct HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1 {};
struct HLT_DoubleMu3_DZ_PFMET50_PFMHT60 {};
struct HLT_DoubleMu3_DZ_PFMET70_PFMHT70 {};
struct HLT_DoubleMu3_DZ_PFMET90_PFMHT90 {};
struct HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass {};
struct HLT_DoubleMu4_Jpsi_Displaced {};
struct HLT_DoubleMu4_Jpsi_NoVertexing {};
struct HLT_DoubleMu4_JpsiTrkTrk_Displaced {};
struct HLT_DoubleMu43NoFiltersNoVtx {};
struct HLT_DoubleMu48NoFiltersNoVtx {};
struct HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL {};
struct HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL {};
struct HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL {};
struct HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL {};
struct HLT_DoubleMu33NoFiltersNoVtxDisplaced {};
struct HLT_DoubleMu40NoFiltersNoVtxDisplaced {};
struct HLT_DoubleMu20_7_Mass0to30_L1_DM4 {};
struct HLT_DoubleMu20_7_Mass0to30_L1_DM4EG {};
struct HLT_HT425 {};
struct HLT_HT430_DisplacedDijet40_DisplacedTrack {};
struct HLT_HT500_DisplacedDijet40_DisplacedTrack {};
struct HLT_HT430_DisplacedDijet60_DisplacedTrack {};
struct HLT_HT400_DisplacedDijet40_DisplacedTrack {};
struct HLT_HT650_DisplacedDijet60_Inclusive {};
struct HLT_HT550_DisplacedDijet60_Inclusive {};
struct HLT_DiJet110_35_Mjj650_PFMET110 {};
struct HLT_DiJet110_35_Mjj650_PFMET120 {};
struct HLT_DiJet110_35_Mjj650_PFMET130 {};
struct HLT_TripleJet110_35_35_Mjj650_PFMET110 {};
struct HLT_TripleJet110_35_35_Mjj650_PFMET120 {};
struct HLT_TripleJet110_35_35_Mjj650_PFMET130 {};
struct HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned {};
struct HLT_Ele28_eta2p1_WPTight_Gsf_HT150 {};
struct HLT_Ele28_HighEta_SC20_Mass55 {};
struct HLT_DoubleMu20_7_Mass0to30_Photon23 {};
struct HLT_Ele15_IsoVVVL_PFHT450_CaloBTagDeepCSV_4p5 {};
struct HLT_Ele15_IsoVVVL_PFHT450_PFMET50 {};
struct HLT_Ele15_IsoVVVL_PFHT450 {};
struct HLT_Ele50_IsoVVVL_PFHT450 {};
struct HLT_Ele15_IsoVVVL_PFHT600 {};
struct HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60 {};
struct HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60 {};
struct HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60 {};
struct HLT_Mu15_IsoVVVL_PFHT450_CaloBTagDeepCSV_4p5 {};
struct HLT_Mu15_IsoVVVL_PFHT450_PFMET50 {};
struct HLT_Mu15_IsoVVVL_PFHT450 {};
struct HLT_Mu50_IsoVVVL_PFHT450 {};
struct HLT_Mu15_IsoVVVL_PFHT600 {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMET70_PFMHT70_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu70_PFMHTNoMu70_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight {};
struct HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight {};
struct HLT_Dimuon10_PsiPrime_Barrel_Seagulls {};
struct HLT_Dimuon20_Jpsi_Barrel_Seagulls {};
struct HLT_Dimuon12_Upsilon_y1p4 {};
struct HLT_Dimuon14_Phi_Barrel_Seagulls {};
struct HLT_Dimuon18_PsiPrime {};
struct HLT_Dimuon25_Jpsi {};
struct HLT_Dimuon18_PsiPrime_noCorrL1 {};
struct HLT_Dimuon24_Upsilon_noCorrL1 {};
struct HLT_Dimuon24_Phi_noCorrL1 {};
struct HLT_Dimuon25_Jpsi_noCorrL1 {};
struct HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8 {};
struct HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ {};
struct HLT_DiMu9_Ele9_CaloIdL_TrackIdL {};
struct HLT_DoubleIsoMu20_eta2p1 {};
struct HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx {};
struct HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx {};
struct HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx {};
struct HLT_Mu8 {};
struct HLT_Mu17 {};
struct HLT_Mu19 {};
struct HLT_Mu17_Photon30_IsoCaloId {};
struct HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30 {};
struct HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30 {};
struct HLT_Ele15_CaloIdL_TrackIdL_IsoVL_PFJet30 {};
struct HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30 {};
struct HLT_Ele8_CaloIdM_TrackIdM_PFJet30 {};
struct HLT_Ele17_CaloIdM_TrackIdM_PFJet30 {};
struct HLT_Ele23_CaloIdM_TrackIdM_PFJet30 {};
struct HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165 {};
struct HLT_Ele115_CaloIdVT_GsfTrkIdT {};
struct HLT_Ele135_CaloIdVT_GsfTrkIdT {};
struct HLT_Ele145_CaloIdVT_GsfTrkIdT {};
struct HLT_Ele200_CaloIdVT_GsfTrkIdT {};
struct HLT_Ele250_CaloIdVT_GsfTrkIdT {};
struct HLT_Ele300_CaloIdVT_GsfTrkIdT {};
struct HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5 {};
struct HLT_PFHT330PT30_QuadPFJet_75_60_45_40 {};
struct HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94 {};
struct HLT_PFHT400_SixPFJet32 {};
struct HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59 {};
struct HLT_PFHT450_SixPFJet36 {};
struct HLT_PFHT350 {};
struct HLT_PFHT350MinPFJet15 {};
struct HLT_Photon60_R9Id90_CaloIdL_IsoL {};
struct HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL {};
struct HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15 {};
struct HLT_ECALHT800 {};
struct HLT_DiSC30_18_EIso_AND_HE_Mass70 {};
struct HLT_Physics {};
struct HLT_Physics_part0 {};
struct HLT_Physics_part1 {};
struct HLT_Physics_part2 {};
struct HLT_Physics_part3 {};
struct HLT_Physics_part4 {};
struct HLT_Physics_part5 {};
struct HLT_Physics_part6 {};
struct HLT_Physics_part7 {};
struct HLT_Random {};
struct HLT_ZeroBias {};
struct HLT_ZeroBias_Alignment {};
struct HLT_ZeroBias_part0 {};
struct HLT_ZeroBias_part1 {};
struct HLT_ZeroBias_part2 {};
struct HLT_ZeroBias_part3 {};
struct HLT_ZeroBias_part4 {};
struct HLT_ZeroBias_part5 {};
struct HLT_ZeroBias_part6 {};
struct HLT_ZeroBias_part7 {};
struct HLT_AK4CaloJet30 {};
struct HLT_AK4CaloJet40 {};
struct HLT_AK4CaloJet50 {};
struct HLT_AK4CaloJet80 {};
struct HLT_AK4CaloJet100 {};
struct HLT_AK4CaloJet120 {};
struct HLT_AK4PFJet30 {};
struct HLT_AK4PFJet50 {};
struct HLT_AK4PFJet80 {};
struct HLT_AK4PFJet100 {};
struct HLT_AK4PFJet120 {};
struct HLT_SinglePhoton10_Eta3p1ForPPRef {};
struct HLT_SinglePhoton20_Eta3p1ForPPRef {};
struct HLT_SinglePhoton30_Eta3p1ForPPRef {};
struct HLT_Photon20_HoverELoose {};
struct HLT_Photon30_HoverELoose {};
struct HLT_EcalCalibration {};
struct HLT_HcalCalibration {};
struct HLT_L1UnpairedBunchBptxMinus {};
struct HLT_L1UnpairedBunchBptxPlus {};
struct HLT_L1NotBptxOR {};
struct HLT_L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142 {};
struct HLT_CDC_L2cosmic_5_er1p0 {};
struct HLT_CDC_L2cosmic_5p5_er1p0 {};
struct HLT_HcalNZS {};
struct HLT_HcalPhiSym {};
struct HLT_HcalIsolatedbunch {};
struct HLT_IsoTrackHB {};
struct HLT_IsoTrackHE {};
struct HLT_ZeroBias_FirstCollisionAfterAbortGap {};
struct HLT_ZeroBias_IsolatedBunches {};
struct HLT_ZeroBias_FirstCollisionInTrain {};
struct HLT_ZeroBias_LastCollisionInTrain {};
struct HLT_ZeroBias_FirstBXAfterTrain {};
struct HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90 {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100 {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET110 {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET120 {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET130 {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET140 {};
struct HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr {};
struct HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr {};
struct HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1 {};
struct HLT_MediumChargedIsoPFTau200HighPtRelaxedIso_Trk50_eta2p1 {};
struct HLT_MediumChargedIsoPFTau220HighPtRelaxedIso_Trk50_eta2p1 {};
struct HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL {};
struct HLT_Rsq0p35 {};
struct HLT_Rsq0p40 {};
struct HLT_RsqMR300_Rsq0p09_MR200 {};
struct HLT_RsqMR320_Rsq0p09_MR200 {};
struct HLT_RsqMR300_Rsq0p09_MR200_4jet {};
struct HLT_RsqMR320_Rsq0p09_MR200_4jet {};
struct HLT_IsoMu27_MET90 {};
struct HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_eta2p1_Reg {};
struct HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg {};
struct HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg {};
struct HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg {};
struct HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1_Reg {};
struct HLT_DoubleTightChargedIsoPFTauHPS40_Trk1_eta2p1_Reg {};
struct HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg {};
struct HLT_DoubleTightChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg {};
struct HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1 {};
struct HLT_VBF_DoubleMediumChargedIsoPFTauHPS20_Trk1_eta2p1 {};
struct HLT_VBF_DoubleTightChargedIsoPFTauHPS20_Trk1_eta2p1 {};
struct HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50 {};
struct HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3 {};
struct HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3 {};
struct HLT_PFMET100_PFMHT100_IDTight_PFHT60 {};
struct HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60 {};
struct HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60 {};
struct HLT_Mu18_Mu9_SameSign {};
struct HLT_Mu18_Mu9_SameSign_DZ {};
struct HLT_Mu18_Mu9 {};
struct HLT_Mu18_Mu9_DZ {};
struct HLT_Mu20_Mu10_SameSign {};
struct HLT_Mu20_Mu10_SameSign_DZ {};
struct HLT_Mu20_Mu10 {};
struct HLT_Mu20_Mu10_DZ {};
struct HLT_Mu23_Mu12_SameSign {};
struct HLT_Mu23_Mu12_SameSign_DZ {};
struct HLT_Mu23_Mu12 {};
struct HLT_Mu23_Mu12_DZ {};
struct HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05 {};
struct HLT_DoubleMu2_Jpsi_DoubleTkMu0_Phi {};
struct HLT_DoubleMu3_DCA_PFMET50_PFMHT60 {};
struct HLT_TripleMu_5_3_3_Mass3p8_DCA {};
struct HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1 {};
struct HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1 {};
struct HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1 {};
struct HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2 {};
struct HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2 {};
struct HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2 {};
struct HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2 {};
struct HLT_QuadPFJet98_83_71_15 {};
struct HLT_QuadPFJet103_88_75_15 {};
struct HLT_QuadPFJet105_88_76_15 {};
struct HLT_QuadPFJet111_90_80_15 {};
struct HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17 {};
struct HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1 {};
struct HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02 {};
struct HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2 {};
struct HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4 {};
struct HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55 {};
struct HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto {};
struct HLT_Mu12_IP6_part0 {};
struct HLT_Mu12_IP6_part1 {};
struct HLT_Mu12_IP6_part2 {};
struct HLT_Mu12_IP6_part3 {};
struct HLT_Mu12_IP6_part4 {};
struct HLT_Mu9_IP5_part0 {};
struct HLT_Mu9_IP5_part1 {};
struct HLT_Mu9_IP5_part2 {};
struct HLT_Mu9_IP5_part3 {};
struct HLT_Mu9_IP5_part4 {};
struct HLT_Mu7_IP4_part0 {};
struct HLT_Mu7_IP4_part1 {};
struct HLT_Mu7_IP4_part2 {};
struct HLT_Mu7_IP4_part3 {};
struct HLT_Mu7_IP4_part4 {};
struct HLT_Mu9_IP4_part0 {};
struct HLT_Mu9_IP4_part1 {};
struct HLT_Mu9_IP4_part2 {};
struct HLT_Mu9_IP4_part3 {};
struct HLT_Mu9_IP4_part4 {};
struct HLT_Mu8_IP5_part0 {};
struct HLT_Mu8_IP5_part1 {};
struct HLT_Mu8_IP5_part2 {};
struct HLT_Mu8_IP5_part3 {};
struct HLT_Mu8_IP5_part4 {};
struct HLT_Mu8_IP6_part0 {};
struct HLT_Mu8_IP6_part1 {};
struct HLT_Mu8_IP6_part2 {};
struct HLT_Mu8_IP6_part3 {};
struct HLT_Mu8_IP6_part4 {};
struct HLT_Mu9_IP6_part0 {};
struct HLT_Mu9_IP6_part1 {};
struct HLT_Mu9_IP6_part2 {};
struct HLT_Mu9_IP6_part3 {};
struct HLT_Mu9_IP6_part4 {};
struct HLT_Mu8_IP3_part0 {};
struct HLT_Mu8_IP3_part1 {};
struct HLT_Mu8_IP3_part2 {};
struct HLT_Mu8_IP3_part3 {};
struct HLT_Mu8_IP3_part4 {};
struct HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1 {};
struct HLT_TrkMu6NoFiltersNoVtx {};
struct HLT_TrkMu16NoFiltersNoVtx {};
struct HLT_DoubleTrkMu_16_6_NoFiltersNoVtx {};
struct HLTriggerFinalPath {};
struct Flag_HBHENoiseFilter {};
struct Flag_HBHENoiseIsoFilter {};
struct Flag_CSCTightHaloFilter {};
struct Flag_CSCTightHaloTrkMuUnvetoFilter {};
struct Flag_CSCTightHalo2015Filter {};
struct Flag_globalTightHalo2016Filter {};
struct Flag_globalSuperTightHalo2016Filter {};
struct Flag_HcalStripHaloFilter {};
struct Flag_hcalLaserEventFilter {};
struct Flag_EcalDeadCellTriggerPrimitiveFilter {};
struct Flag_EcalDeadCellBoundaryEnergyFilter {};
struct Flag_ecalBadCalibFilter {};
struct Flag_goodVertices {};
struct Flag_eeBadScFilter {};
struct Flag_ecalLaserCorrFilter {};
struct Flag_trkPOGFilters {};
struct Flag_chargedHadronTrackResolutionFilter {};
struct Flag_muonBadTrackFilter {};
struct Flag_BadChargedCandidateFilter {};
struct Flag_BadPFMuonFilter {};
struct Flag_BadChargedCandidateSummer16Filter {};
struct Flag_BadPFMuonSummer16Filter {};
struct Flag_trkPOG_manystripclus53X {};
struct Flag_trkPOG_toomanystripclus53X {};
struct Flag_trkPOG_logErrorTooManyClusters {};
struct Flag_METFilters {};
struct L1Reco_step {};
struct L1_AlwaysTrue {};
struct L1_BPTX_AND_Ref1_VME {};
struct L1_BPTX_AND_Ref3_VME {};
struct L1_BPTX_AND_Ref4_VME {};
struct L1_BPTX_BeamGas_B1_VME {};
struct L1_BPTX_BeamGas_B2_VME {};
struct L1_BPTX_BeamGas_Ref1_VME {};
struct L1_BPTX_BeamGas_Ref2_VME {};
struct L1_BPTX_NotOR_VME {};
struct L1_BPTX_OR_Ref3_VME {};
struct L1_BPTX_OR_Ref4_VME {};
struct L1_BPTX_RefAND_VME {};
struct L1_BptxMinus {};
struct L1_BptxOR {};
struct L1_BptxPlus {};
struct L1_BptxXOR {};
struct L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142 {};
struct L1_DoubleEG8er2p5_HTT260er {};
struct L1_DoubleEG8er2p5_HTT280er {};
struct L1_DoubleEG8er2p5_HTT300er {};
struct L1_DoubleEG8er2p5_HTT320er {};
struct L1_DoubleEG8er2p5_HTT340er {};
struct L1_DoubleEG_15_10_er2p5 {};
struct L1_DoubleEG_20_10_er2p5 {};
struct L1_DoubleEG_22_10_er2p5 {};
struct L1_DoubleEG_25_12_er2p5 {};
struct L1_DoubleEG_25_14_er2p5 {};
struct L1_DoubleEG_27_14_er2p5 {};
struct L1_DoubleEG_LooseIso20_10_er2p5 {};
struct L1_DoubleEG_LooseIso22_10_er2p5 {};
struct L1_DoubleEG_LooseIso22_12_er2p5 {};
struct L1_DoubleEG_LooseIso25_12_er2p5 {};
struct L1_DoubleIsoTau32er2p1 {};
struct L1_DoubleIsoTau34er2p1 {};
struct L1_DoubleIsoTau36er2p1 {};
struct L1_DoubleJet100er2p3_dEta_Max1p6 {};
struct L1_DoubleJet100er2p5 {};
struct L1_DoubleJet112er2p3_dEta_Max1p6 {};
struct L1_DoubleJet120er2p5 {};
struct L1_DoubleJet150er2p5 {};
struct L1_DoubleJet30er2p5_Mass_Min150_dEta_Max1p5 {};
struct L1_DoubleJet30er2p5_Mass_Min200_dEta_Max1p5 {};
struct L1_DoubleJet30er2p5_Mass_Min250_dEta_Max1p5 {};
struct L1_DoubleJet30er2p5_Mass_Min300_dEta_Max1p5 {};
struct L1_DoubleJet30er2p5_Mass_Min330_dEta_Max1p5 {};
struct L1_DoubleJet30er2p5_Mass_Min360_dEta_Max1p5 {};
struct L1_DoubleJet35_Mass_Min450_IsoTau45_RmOvlp {};
struct L1_DoubleJet40er2p5 {};
struct L1_DoubleJet_100_30_DoubleJet30_Mass_Min620 {};
struct L1_DoubleJet_110_35_DoubleJet35_Mass_Min620 {};
struct L1_DoubleJet_115_40_DoubleJet40_Mass_Min620 {};
struct L1_DoubleJet_115_40_DoubleJet40_Mass_Min620_Jet60TT28 {};
struct L1_DoubleJet_120_45_DoubleJet45_Mass_Min620 {};
struct L1_DoubleJet_120_45_DoubleJet45_Mass_Min620_Jet60TT28 {};
struct L1_DoubleJet_80_30_Mass_Min420_DoubleMu0_SQ {};
struct L1_DoubleJet_80_30_Mass_Min420_IsoTau40_RmOvlp {};
struct L1_DoubleJet_80_30_Mass_Min420_Mu8 {};
struct L1_DoubleJet_90_30_DoubleJet30_Mass_Min620 {};
struct L1_DoubleLooseIsoEG22er2p1 {};
struct L1_DoubleLooseIsoEG24er2p1 {};
struct L1_DoubleMu0 {};
struct L1_DoubleMu0_Mass_Min1 {};
struct L1_DoubleMu0_OQ {};
struct L1_DoubleMu0_SQ {};
struct L1_DoubleMu0_SQ_OS {};
struct L1_DoubleMu0_dR_Max1p6_Jet90er2p5_dR_Max0p8 {};
struct L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 {};
struct L1_DoubleMu0er1p5_SQ {};
struct L1_DoubleMu0er1p5_SQ_OS {};
struct L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 {};
struct L1_DoubleMu0er1p5_SQ_dR_Max1p4 {};
struct L1_DoubleMu0er2p0_SQ_OS_dR_Max1p4 {};
struct L1_DoubleMu0er2p0_SQ_dR_Max1p4 {};
struct L1_DoubleMu10_SQ {};
struct L1_DoubleMu18er2p1 {};
struct L1_DoubleMu3_OS_DoubleEG7p5Upsilon {};
struct L1_DoubleMu3_SQ_ETMHF50_HTT60er {};
struct L1_DoubleMu3_SQ_ETMHF50_Jet60er2p5 {};
struct L1_DoubleMu3_SQ_ETMHF50_Jet60er2p5_OR_DoubleJet40er2p5 {};
struct L1_DoubleMu3_SQ_ETMHF60_Jet60er2p5 {};
struct L1_DoubleMu3_SQ_HTT220er {};
struct L1_DoubleMu3_SQ_HTT240er {};
struct L1_DoubleMu3_SQ_HTT260er {};
struct L1_DoubleMu3_dR_Max1p6_Jet90er2p5_dR_Max0p8 {};
struct L1_DoubleMu4_SQ_EG9er2p5 {};
struct L1_DoubleMu4_SQ_OS {};
struct L1_DoubleMu4_SQ_OS_dR_Max1p2 {};
struct L1_DoubleMu4p5_SQ_OS {};
struct L1_DoubleMu4p5_SQ_OS_dR_Max1p2 {};
struct L1_DoubleMu4p5er2p0_SQ_OS {};
struct L1_DoubleMu4p5er2p0_SQ_OS_Mass7to18 {};
struct L1_DoubleMu5Upsilon_OS_DoubleEG3 {};
struct L1_DoubleMu5_SQ_EG9er2p5 {};
struct L1_DoubleMu9_SQ {};
struct L1_DoubleMu_12_5 {};
struct L1_DoubleMu_15_5_SQ {};
struct L1_DoubleMu_15_7 {};
struct L1_DoubleMu_15_7_Mass_Min1 {};
struct L1_DoubleMu_15_7_SQ {};
struct L1_DoubleTau70er2p1 {};
struct L1_ETM120 {};
struct L1_ETM150 {};
struct L1_ETMHF100 {};
struct L1_ETMHF100_HTT60er {};
struct L1_ETMHF110 {};
struct L1_ETMHF110_HTT60er {};
struct L1_ETMHF110_HTT60er_NotSecondBunchInTrain {};
struct L1_ETMHF120 {};
struct L1_ETMHF120_HTT60er {};
struct L1_ETMHF120_NotSecondBunchInTrain {};
struct L1_ETMHF130 {};
struct L1_ETMHF130_HTT60er {};
struct L1_ETMHF140 {};
struct L1_ETMHF150 {};
struct L1_ETMHF90_HTT60er {};
struct L1_ETT1200 {};
struct L1_ETT1600 {};
struct L1_ETT2000 {};
struct L1_FirstBunchAfterTrain {};
struct L1_FirstBunchBeforeTrain {};
struct L1_FirstBunchInTrain {};
struct L1_FirstCollisionInOrbit {};
struct L1_FirstCollisionInTrain {};
struct L1_HCAL_LaserMon_Trig {};
struct L1_HCAL_LaserMon_Veto {};
struct L1_HTT120er {};
struct L1_HTT160er {};
struct L1_HTT200er {};
struct L1_HTT255er {};
struct L1_HTT280er {};
struct L1_HTT280er_QuadJet_70_55_40_35_er2p4 {};
struct L1_HTT320er {};
struct L1_HTT320er_QuadJet_70_55_40_40_er2p4 {};
struct L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3 {};
struct L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3 {};
struct L1_HTT360er {};
struct L1_HTT400er {};
struct L1_HTT450er {};
struct L1_IsoEG32er2p5_Mt40 {};
struct L1_IsoEG32er2p5_Mt44 {};
struct L1_IsoEG32er2p5_Mt48 {};
struct L1_IsoTau40er2p1_ETMHF100 {};
struct L1_IsoTau40er2p1_ETMHF110 {};
struct L1_IsoTau40er2p1_ETMHF120 {};
struct L1_IsoTau40er2p1_ETMHF90 {};
struct L1_IsolatedBunch {};
struct L1_LastBunchInTrain {};
struct L1_LastCollisionInTrain {};
struct L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3 {};
struct L1_LooseIsoEG22er2p1_Tau70er2p1_dR_Min0p3 {};
struct L1_LooseIsoEG24er2p1_HTT100er {};
struct L1_LooseIsoEG24er2p1_IsoTau27er2p1_dR_Min0p3 {};
struct L1_LooseIsoEG26er2p1_HTT100er {};
struct L1_LooseIsoEG26er2p1_Jet34er2p5_dR_Min0p3 {};
struct L1_LooseIsoEG28er2p1_HTT100er {};
struct L1_LooseIsoEG28er2p1_Jet34er2p5_dR_Min0p3 {};
struct L1_LooseIsoEG30er2p1_HTT100er {};
struct L1_LooseIsoEG30er2p1_Jet34er2p5_dR_Min0p3 {};
struct L1_MinimumBiasHF0_AND_BptxAND {};
struct L1_Mu10er2p3_Jet32er2p3_dR_Max0p4_DoubleJet32er2p3_dEta_Max1p6 {};
struct L1_Mu12er2p3_Jet40er2p1_dR_Max0p4_DoubleJet40er2p1_dEta_Max1p6 {};
struct L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6 {};
struct L1_Mu18er2p1_Tau24er2p1 {};
struct L1_Mu18er2p1_Tau26er2p1 {};
struct L1_Mu20_EG10er2p5 {};
struct L1_Mu22er2p1_IsoTau32er2p1 {};
struct L1_Mu22er2p1_IsoTau34er2p1 {};
struct L1_Mu22er2p1_IsoTau36er2p1 {};
struct L1_Mu22er2p1_IsoTau40er2p1 {};
struct L1_Mu22er2p1_Tau70er2p1 {};
struct L1_Mu3_Jet120er2p5_dR_Max0p4 {};
struct L1_Mu3_Jet120er2p5_dR_Max0p8 {};
struct L1_Mu3_Jet16er2p5_dR_Max0p4 {};
struct L1_Mu3_Jet30er2p5 {};
struct L1_Mu3_Jet35er2p5_dR_Max0p4 {};
struct L1_Mu3_Jet60er2p5_dR_Max0p4 {};
struct L1_Mu3_Jet80er2p5_dR_Max0p4 {};
struct L1_Mu3er1p5_Jet100er2p5_ETMHF40 {};
struct L1_Mu3er1p5_Jet100er2p5_ETMHF50 {};
struct L1_Mu5_EG23er2p5 {};
struct L1_Mu5_LooseIsoEG20er2p5 {};
struct L1_Mu6_DoubleEG10er2p5 {};
struct L1_Mu6_DoubleEG12er2p5 {};
struct L1_Mu6_DoubleEG15er2p5 {};
struct L1_Mu6_DoubleEG17er2p5 {};
struct L1_Mu6_HTT240er {};
struct L1_Mu6_HTT250er {};
struct L1_Mu7_EG23er2p5 {};
struct L1_Mu7_LooseIsoEG20er2p5 {};
struct L1_Mu7_LooseIsoEG23er2p5 {};
struct L1_NotBptxOR {};
struct L1_QuadJet36er2p5_IsoTau52er2p1 {};
struct L1_QuadJet60er2p5 {};
struct L1_QuadJet_95_75_65_20_DoubleJet_75_65_er2p5_Jet20_FWD3p0 {};
struct L1_QuadMu0 {};
struct L1_QuadMu0_OQ {};
struct L1_QuadMu0_SQ {};
struct L1_SecondBunchInTrain {};
struct L1_SecondLastBunchInTrain {};
struct L1_SingleEG10er2p5 {};
struct L1_SingleEG15er2p5 {};
struct L1_SingleEG26er2p5 {};
struct L1_SingleEG34er2p5 {};
struct L1_SingleEG36er2p5 {};
struct L1_SingleEG38er2p5 {};
struct L1_SingleEG40er2p5 {};
struct L1_SingleEG42er2p5 {};
struct L1_SingleEG45er2p5 {};
struct L1_SingleEG50 {};
struct L1_SingleEG60 {};
struct L1_SingleEG8er2p5 {};
struct L1_SingleIsoEG24er1p5 {};
struct L1_SingleIsoEG24er2p1 {};
struct L1_SingleIsoEG26er1p5 {};
struct L1_SingleIsoEG26er2p1 {};
struct L1_SingleIsoEG26er2p5 {};
struct L1_SingleIsoEG28er1p5 {};
struct L1_SingleIsoEG28er2p1 {};
struct L1_SingleIsoEG28er2p5 {};
struct L1_SingleIsoEG30er2p1 {};
struct L1_SingleIsoEG30er2p5 {};
struct L1_SingleIsoEG32er2p1 {};
struct L1_SingleIsoEG32er2p5 {};
struct L1_SingleIsoEG34er2p5 {};
struct L1_SingleJet10erHE {};
struct L1_SingleJet120 {};
struct L1_SingleJet120_FWD3p0 {};
struct L1_SingleJet120er2p5 {};
struct L1_SingleJet12erHE {};
struct L1_SingleJet140er2p5 {};
struct L1_SingleJet140er2p5_ETMHF80 {};
struct L1_SingleJet140er2p5_ETMHF90 {};
struct L1_SingleJet160er2p5 {};
struct L1_SingleJet180 {};
struct L1_SingleJet180er2p5 {};
struct L1_SingleJet200 {};
struct L1_SingleJet20er2p5_NotBptxOR {};
struct L1_SingleJet20er2p5_NotBptxOR_3BX {};
struct L1_SingleJet35 {};
struct L1_SingleJet35_FWD3p0 {};
struct L1_SingleJet35er2p5 {};
struct L1_SingleJet43er2p5_NotBptxOR_3BX {};
struct L1_SingleJet46er2p5_NotBptxOR_3BX {};
struct L1_SingleJet60 {};
struct L1_SingleJet60_FWD3p0 {};
struct L1_SingleJet60er2p5 {};
struct L1_SingleJet8erHE {};
struct L1_SingleJet90 {};
struct L1_SingleJet90_FWD3p0 {};
struct L1_SingleJet90er2p5 {};
struct L1_SingleLooseIsoEG28er1p5 {};
struct L1_SingleLooseIsoEG30er1p5 {};
struct L1_SingleMu0_BMTF {};
struct L1_SingleMu0_DQ {};
struct L1_SingleMu0_EMTF {};
struct L1_SingleMu0_OMTF {};
struct L1_SingleMu10er1p5 {};
struct L1_SingleMu12_DQ_BMTF {};
struct L1_SingleMu12_DQ_EMTF {};
struct L1_SingleMu12_DQ_OMTF {};
struct L1_SingleMu12er1p5 {};
struct L1_SingleMu14er1p5 {};
struct L1_SingleMu15_DQ {};
struct L1_SingleMu16er1p5 {};
struct L1_SingleMu18 {};
struct L1_SingleMu18er1p5 {};
struct L1_SingleMu20 {};
struct L1_SingleMu22 {};
struct L1_SingleMu22_BMTF {};
struct L1_SingleMu22_EMTF {};
struct L1_SingleMu22_OMTF {};
struct L1_SingleMu25 {};
struct L1_SingleMu3 {};
struct L1_SingleMu5 {};
struct L1_SingleMu6er1p5 {};
struct L1_SingleMu7 {};
struct L1_SingleMu7_DQ {};
struct L1_SingleMu7er1p5 {};
struct L1_SingleMu8er1p5 {};
struct L1_SingleMu9er1p5 {};
struct L1_SingleMuCosmics {};
struct L1_SingleMuCosmics_BMTF {};
struct L1_SingleMuCosmics_EMTF {};
struct L1_SingleMuCosmics_OMTF {};
struct L1_SingleMuOpen {};
struct L1_SingleMuOpen_NotBptxOR {};
struct L1_SingleMuOpen_er1p1_NotBptxOR_3BX {};
struct L1_SingleMuOpen_er1p4_NotBptxOR_3BX {};
struct L1_SingleTau120er2p1 {};
struct L1_SingleTau130er2p1 {};
struct L1_TOTEM_1 {};
struct L1_TOTEM_2 {};
struct L1_TOTEM_3 {};
struct L1_TOTEM_4 {};
struct L1_TripleEG16er2p5 {};
struct L1_TripleEG_16_12_8_er2p5 {};
struct L1_TripleEG_16_15_8_er2p5 {};
struct L1_TripleEG_18_17_8_er2p5 {};
struct L1_TripleEG_18_18_12_er2p5 {};
struct L1_TripleJet_100_80_70_DoubleJet_80_70_er2p5 {};
struct L1_TripleJet_105_85_75_DoubleJet_85_75_er2p5 {};
struct L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5 {};
struct L1_TripleMu0 {};
struct L1_TripleMu0_OQ {};
struct L1_TripleMu0_SQ {};
struct L1_TripleMu3 {};
struct L1_TripleMu3_SQ {};
struct L1_TripleMu_5SQ_3SQ_0OQ {};
struct L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu_5_3_SQ_OS_Mass_Max9 {};
struct L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9 {};
struct L1_TripleMu_5_3_3 {};
struct L1_TripleMu_5_3_3_SQ {};
struct L1_TripleMu_5_3p5_2p5 {};
struct L1_TripleMu_5_3p5_2p5_DoubleMu_5_2p5_OS_Mass_5to17 {};
struct L1_TripleMu_5_3p5_2p5_OQ_DoubleMu_5_2p5_OQ_OS_Mass_5to17 {};
struct L1_TripleMu_5_4_2p5_DoubleMu_5_2p5_OS_Mass_5to17 {};
struct L1_TripleMu_5_5_3 {};
struct L1_UnpairedBunchBptxMinus {};
struct L1_UnpairedBunchBptxPlus {};
struct L1_ZeroBias {};
struct L1_ZeroBias_copy {};

using CorrT1METJet = llama::Record<
    llama::Field<CorrT1METJet_area, float>,
    llama::Field<CorrT1METJet_eta, float>,
    llama::Field<CorrT1METJet_muonSubtrFactor, float>,
    llama::Field<CorrT1METJet_phi, float>,
    llama::Field<CorrT1METJet_rawPt, float>
>;

using Electron = llama::Record<
    llama::Field<Electron_deltaEtaSC, float>,
    llama::Field<Electron_dr03EcalRecHitSumEt, float>,
    llama::Field<Electron_dr03HcalDepth1TowerSumEt, float>,
    llama::Field<Electron_dr03TkSumPt, float>,
    llama::Field<Electron_dr03TkSumPtHEEP, float>,
    llama::Field<Electron_dxy, float>,
    llama::Field<Electron_dxyErr, float>,
    llama::Field<Electron_dz, float>,
    llama::Field<Electron_dzErr, float>,
    llama::Field<Electron_eCorr, float>,
    llama::Field<Electron_eInvMinusPInv, float>,
    llama::Field<Electron_energyErr, float>,
    llama::Field<Electron_eta, float>,
    llama::Field<Electron_hoe, float>,
    llama::Field<Electron_ip3d, float>,
    llama::Field<Electron_jetPtRelv2, float>,
    llama::Field<Electron_jetRelIso, float>,
    llama::Field<Electron_mass, float>,
    llama::Field<Electron_miniPFRelIso_all, float>,
    llama::Field<Electron_miniPFRelIso_chg, float>,
    llama::Field<Electron_mvaFall17V1Iso, float>,
    llama::Field<Electron_mvaFall17V1noIso, float>,
    llama::Field<Electron_mvaFall17V2Iso, float>,
    llama::Field<Electron_mvaFall17V2noIso, float>,
    llama::Field<Electron_pfRelIso03_all, float>,
    llama::Field<Electron_pfRelIso03_chg, float>,
    llama::Field<Electron_phi, float>,
    llama::Field<Electron_pt, float>,
    llama::Field<Electron_r9, float>,
    llama::Field<Electron_sieie, float>,
    llama::Field<Electron_sip3d, float>,
    llama::Field<Electron_mvaTTH, float>,
    llama::Field<Electron_charge, std::int32_t>,
    llama::Field<Electron_cutBased, std::int32_t>,
    llama::Field<Electron_cutBased_Fall17_V1, std::int32_t>,
    llama::Field<Electron_jetIdx, std::int32_t>,
    llama::Field<Electron_pdgId, std::int32_t>,
    llama::Field<Electron_photonIdx, std::int32_t>,
    llama::Field<Electron_tightCharge, std::int32_t>,
    //llama::Field<Electron_vidNestedWPbitmap, std::int32_t>,
    llama::Field<Electron_convVeto, bit>,
    llama::Field<Electron_cutBased_HEEP, bit>,
    llama::Field<Electron_isPFcand, bit>,
    llama::Field<Electron_lostHits, byte>,
    llama::Field<Electron_mvaFall17V1Iso_WP80, bit>,
    llama::Field<Electron_mvaFall17V1Iso_WP90, bit>,
    llama::Field<Electron_mvaFall17V1Iso_WPL, bit>,
    llama::Field<Electron_mvaFall17V1noIso_WP80, bit>,
    llama::Field<Electron_mvaFall17V1noIso_WP90, bit>,
    llama::Field<Electron_mvaFall17V1noIso_WPL, bit>,
    llama::Field<Electron_mvaFall17V2Iso_WP80, bit>,
    llama::Field<Electron_mvaFall17V2Iso_WP90, bit>,
    llama::Field<Electron_mvaFall17V2Iso_WPL, bit>,
    llama::Field<Electron_mvaFall17V2noIso_WP80, bit>,
    llama::Field<Electron_mvaFall17V2noIso_WP90, bit>,
    llama::Field<Electron_mvaFall17V2noIso_WPL, bit>,
    llama::Field<Electron_seedGain, byte>,
    llama::Field<Electron_genPartIdx, std::int32_t>,
    llama::Field<Electron_genPartFlav, byte>,
    llama::Field<Electron_cleanmask, byte>
>;

using FatJet = llama::Record<
    llama::Field<FatJet_area, float>,
    llama::Field<FatJet_btagCMVA, float>,
    llama::Field<FatJet_btagCSVV2, float>,
    llama::Field<FatJet_btagDDBvL, float>,
    llama::Field<FatJet_btagDDCvB, float>,
    llama::Field<FatJet_btagDDCvL, float>,
    llama::Field<FatJet_btagDeepB, float>,
    llama::Field<FatJet_btagHbb, float>,
    llama::Field<FatJet_deepTagMD_H4qvsQCD, float>,
    llama::Field<FatJet_deepTagMD_HbbvsQCD, float>,
    llama::Field<FatJet_deepTagMD_TvsQCD, float>,
    llama::Field<FatJet_deepTagMD_WvsQCD, float>,
    llama::Field<FatJet_deepTagMD_ZHbbvsQCD, float>,
    llama::Field<FatJet_deepTagMD_ZHccvsQCD, float>,
    llama::Field<FatJet_deepTagMD_ZbbvsQCD, float>,
    llama::Field<FatJet_deepTagMD_ZvsQCD, float>,
    llama::Field<FatJet_deepTagMD_bbvsLight, float>,
    llama::Field<FatJet_deepTagMD_ccvsLight, float>,
    llama::Field<FatJet_deepTag_H, float>,
    llama::Field<FatJet_deepTag_QCD, float>,
    llama::Field<FatJet_deepTag_QCDothers, float>,
    llama::Field<FatJet_deepTag_TvsQCD, float>,
    llama::Field<FatJet_deepTag_WvsQCD, float>,
    llama::Field<FatJet_deepTag_ZvsQCD, float>,
    llama::Field<FatJet_eta, float>,
    llama::Field<FatJet_mass, float>,
    llama::Field<FatJet_msoftdrop, float>,
    llama::Field<FatJet_n2b1, float>,
    llama::Field<FatJet_n3b1, float>,
    llama::Field<FatJet_phi, float>,
    llama::Field<FatJet_pt, float>,
    llama::Field<FatJet_rawFactor, float>,
    llama::Field<FatJet_tau1, float>,
    llama::Field<FatJet_tau2, float>,
    llama::Field<FatJet_tau3, float>,
    llama::Field<FatJet_tau4, float>,
    llama::Field<FatJet_jetId, std::int32_t>,
    llama::Field<FatJet_subJetIdx1, std::int32_t>,
    llama::Field<FatJet_subJetIdx2, std::int32_t>
>;

using GenJetAK8 = llama::Record<
    llama::Field<GenJetAK8_eta, float>,
    llama::Field<GenJetAK8_mass, float>,
    llama::Field<GenJetAK8_phi, float>,
    llama::Field<GenJetAK8_pt, float>,
    llama::Field<GenJetAK8_partonFlavour, std::int32_t>,
    llama::Field<GenJetAK8_hadronFlavour, byte>
>;

using GenJet = llama::Record<
    llama::Field<GenJet_eta, float>,
    llama::Field<GenJet_mass, float>,
    llama::Field<GenJet_phi, float>,
    llama::Field<GenJet_pt, float>,
    llama::Field<GenJet_partonFlavour, std::int32_t>,
    llama::Field<GenJet_hadronFlavour, byte>
>;

using GenPart = llama::Record<
    llama::Field<GenPart_eta, float>,
    llama::Field<GenPart_mass, float>,
    llama::Field<GenPart_phi, float>,
    llama::Field<GenPart_pt, float>,
    llama::Field<GenPart_genPartIdxMother, std::int32_t>,
    llama::Field<GenPart_pdgId, std::int32_t>,
    llama::Field<GenPart_status, std::int32_t>,
    llama::Field<GenPart_statusFlags, std::int32_t>
>;

using SubGenJetAK8 = llama::Record<
    llama::Field<SubGenJetAK8_eta, float>,
    llama::Field<SubGenJetAK8_mass, float>,
    llama::Field<SubGenJetAK8_phi, float>,
    llama::Field<SubGenJetAK8_pt, float>
>;

using GenVisTau = llama::Record<
    llama::Field<GenVisTau_eta, float>,
    llama::Field<GenVisTau_mass, float>,
    llama::Field<GenVisTau_phi, float>,
    llama::Field<GenVisTau_pt, float>,
    llama::Field<GenVisTau_charge, std::int32_t>,
    llama::Field<GenVisTau_genPartIdxMother, std::int32_t>,
    llama::Field<GenVisTau_status, std::int32_t>
>;

using LHEPdfWeight_ = llama::Record<
    llama::Field<LHEPdfWeight, float>
>;

using LHEReweightingWeight_ = llama::Record<
    llama::Field<LHEReweightingWeight, float>
>;

using LHEScaleWeight_ = llama::Record<
    llama::Field<LHEScaleWeight, float>
>;

using PSWeight_ = llama::Record<
    llama::Field<PSWeight, float>
>;

using IsoTrack = llama::Record<
    llama::Field<IsoTrack_dxy, float>,
    llama::Field<IsoTrack_dz, float>,
    llama::Field<IsoTrack_eta, float>,
    llama::Field<IsoTrack_pfRelIso03_all, float>,
    llama::Field<IsoTrack_pfRelIso03_chg, float>,
    llama::Field<IsoTrack_phi, float>,
    llama::Field<IsoTrack_pt, float>,
    llama::Field<IsoTrack_miniPFRelIso_all, float>,
    llama::Field<IsoTrack_miniPFRelIso_chg, float>,
    llama::Field<IsoTrack_fromPV, std::int32_t>,
    llama::Field<IsoTrack_pdgId, std::int32_t>,
    llama::Field<IsoTrack_isHighPurityTrack, bit>,
    llama::Field<IsoTrack_isPFcand, bit>,
    llama::Field<IsoTrack_isFromLostTrack, bit>
>;

using Jet = llama::Record<
    llama::Field<Jet_area, float>,
    llama::Field<Jet_btagCMVA, float>,
    llama::Field<Jet_btagCSVV2, float>,
    llama::Field<Jet_btagDeepB, float>,
    llama::Field<Jet_btagDeepC, float>,
    llama::Field<Jet_btagDeepFlavB, float>,
    llama::Field<Jet_btagDeepFlavC, float>,
    llama::Field<Jet_chEmEF, float>,
    llama::Field<Jet_chHEF, float>,
    llama::Field<Jet_eta, float>,
    llama::Field<Jet_jercCHF, float>,
    llama::Field<Jet_jercCHPUF, float>,
    llama::Field<Jet_mass, float>,
    llama::Field<Jet_muEF, float>,
    llama::Field<Jet_muonSubtrFactor, float>,
    llama::Field<Jet_neEmEF, float>,
    llama::Field<Jet_neHEF, float>,
    llama::Field<Jet_phi, float>,
    llama::Field<Jet_pt, float>,
    llama::Field<Jet_qgl, float>,
    llama::Field<Jet_rawFactor, float>,
    llama::Field<Jet_bRegCorr, float>,
    llama::Field<Jet_bRegRes, float>,
    llama::Field<Jet_electronIdx1, std::int32_t>,
    llama::Field<Jet_electronIdx2, std::int32_t>,
    llama::Field<Jet_jetId, std::int32_t>,
    llama::Field<Jet_muonIdx1, std::int32_t>,
    llama::Field<Jet_muonIdx2, std::int32_t>,
    llama::Field<Jet_nConstituents, std::int32_t>,
    llama::Field<Jet_nElectrons, std::int32_t>,
    llama::Field<Jet_nMuons, std::int32_t>,
    llama::Field<Jet_puId, std::int32_t>,
    llama::Field<Jet_genJetIdx, std::int32_t>,
    llama::Field<Jet_hadronFlavour, std::int32_t>,
    llama::Field<Jet_partonFlavour, std::int32_t>,
    llama::Field<Jet_cleanmask, byte>
>;

using LHEPart = llama::Record<
    llama::Field<LHEPart_pt, float>,
    llama::Field<LHEPart_eta, float>,
    llama::Field<LHEPart_phi, float>,
    llama::Field<LHEPart_mass, float>,
    llama::Field<LHEPart_pdgId, std::int32_t>
>;

using Muon = llama::Record<
    llama::Field<Muon_dxy, float>,
    llama::Field<Muon_dxyErr, float>,
    llama::Field<Muon_dz, float>,
    llama::Field<Muon_dzErr, float>,
    llama::Field<Muon_eta, float>,
    llama::Field<Muon_ip3d, float>,
    llama::Field<Muon_jetPtRelv2, float>,
    llama::Field<Muon_jetRelIso, float>,
    llama::Field<Muon_mass, float>,
    llama::Field<Muon_miniPFRelIso_all, float>,
    llama::Field<Muon_miniPFRelIso_chg, float>,
    llama::Field<Muon_pfRelIso03_all, float>,
    llama::Field<Muon_pfRelIso03_chg, float>,
    llama::Field<Muon_pfRelIso04_all, float>,
    llama::Field<Muon_phi, float>,
    llama::Field<Muon_pt, float>,
    llama::Field<Muon_ptErr, float>,
    llama::Field<Muon_segmentComp, float>,
    llama::Field<Muon_sip3d, float>,
    llama::Field<Muon_softMva, float>,
    llama::Field<Muon_tkRelIso, float>,
    llama::Field<Muon_tunepRelPt, float>,
    llama::Field<Muon_mvaLowPt, float>,
    llama::Field<Muon_mvaTTH, float>,
    llama::Field<Muon_charge, std::int32_t>,
    llama::Field<Muon_jetIdx, std::int32_t>,
    llama::Field<Muon_nStations, std::int32_t>,
    llama::Field<Muon_nTrackerLayers, std::int32_t>,
    llama::Field<Muon_pdgId, std::int32_t>,
    llama::Field<Muon_tightCharge, std::int32_t>,
    llama::Field<Muon_highPtId, byte>,
    llama::Field<Muon_inTimeMuon, bit>,
    llama::Field<Muon_isGlobal, bit>,
    llama::Field<Muon_isPFcand, bit>,
    llama::Field<Muon_isTracker, bit>,
    llama::Field<Muon_looseId, bit>,
    llama::Field<Muon_mediumId, bit>,
    llama::Field<Muon_mediumPromptId, bit>,
    llama::Field<Muon_miniIsoId, byte>,
    llama::Field<Muon_multiIsoId, byte>,
    llama::Field<Muon_mvaId, byte>,
    llama::Field<Muon_pfIsoId, byte>,
    llama::Field<Muon_softId, bit>,
    llama::Field<Muon_softMvaId, bit>,
    llama::Field<Muon_tightId, bit>,
    llama::Field<Muon_tkIsoId, byte>,
    llama::Field<Muon_triggerIdLoose, bit>,
    llama::Field<Muon_genPartIdx, std::int32_t>,
    llama::Field<Muon_genPartFlav, byte>,
    llama::Field<Muon_cleanmask, byte>
>;

using Photon = llama::Record<
    llama::Field<Photon_eCorr, float>,
    llama::Field<Photon_energyErr, float>,
    llama::Field<Photon_eta, float>,
    llama::Field<Photon_hoe, float>,
    llama::Field<Photon_mass, float>,
    llama::Field<Photon_mvaID, float>,
    llama::Field<Photon_mvaIDV1, float>,
    llama::Field<Photon_pfRelIso03_all, float>,
    llama::Field<Photon_pfRelIso03_chg, float>,
    llama::Field<Photon_phi, float>,
    llama::Field<Photon_pt, float>,
    llama::Field<Photon_r9, float>,
    llama::Field<Photon_sieie, float>,
    llama::Field<Photon_charge, std::int32_t>,
    llama::Field<Photon_cutBasedbitmap, std::int32_t>,
    llama::Field<Photon_cutBasedV1bitmap, std::int32_t>,
    llama::Field<Photon_electronIdx, std::int32_t>,
    llama::Field<Photon_jetIdx, std::int32_t>,
    llama::Field<Photon_pdgId, std::int32_t>,
    llama::Field<Photon_vidNestedWPbitmap, std::int32_t>,
    llama::Field<Photon_electronVeto, bit>,
    llama::Field<Photon_isScEtaEB, bit>,
    llama::Field<Photon_isScEtaEE, bit>,
    llama::Field<Photon_mvaID_WP80, bit>,
    llama::Field<Photon_mvaID_WP90, bit>,
    llama::Field<Photon_pixelSeed, bit>,
    llama::Field<Photon_seedGain, byte>,
    llama::Field<Photon_genPartIdx, std::int32_t>,
    llama::Field<Photon_genPartFlav, byte>,
    llama::Field<Photon_cleanmask, byte>
>;

using GenDressedLepton = llama::Record<
    llama::Field<GenDressedLepton_eta, float>,
    llama::Field<GenDressedLepton_mass, float>,
    llama::Field<GenDressedLepton_phi, float>,
    llama::Field<GenDressedLepton_pt, float>,
    llama::Field<GenDressedLepton_pdgId, std::int32_t>,
    llama::Field<GenDressedLepton_hasTauAnc, bit>
>;

using SoftActivityJet = llama::Record<
    llama::Field<SoftActivityJet_eta, float>,
    llama::Field<SoftActivityJet_phi, float>,
    llama::Field<SoftActivityJet_pt, float>
>;

using SubJet = llama::Record<
    llama::Field<SubJet_btagCMVA, float>,
    llama::Field<SubJet_btagCSVV2, float>,
    llama::Field<SubJet_btagDeepB, float>,
    llama::Field<SubJet_eta, float>,
    llama::Field<SubJet_mass, float>,
    llama::Field<SubJet_n2b1, float>,
    llama::Field<SubJet_n3b1, float>,
    llama::Field<SubJet_phi, float>,
    llama::Field<SubJet_pt, float>,
    llama::Field<SubJet_rawFactor, float>,
    llama::Field<SubJet_tau1, float>,
    llama::Field<SubJet_tau2, float>,
    llama::Field<SubJet_tau3, float>,
    llama::Field<SubJet_tau4, float>
>;

using Tau = llama::Record<
    llama::Field<Tau_chargedIso, float>,
    llama::Field<Tau_dxy, float>,
    llama::Field<Tau_dz, float>,
    llama::Field<Tau_eta, float>,
    llama::Field<Tau_leadTkDeltaEta, float>,
    llama::Field<Tau_leadTkDeltaPhi, float>,
    llama::Field<Tau_leadTkPtOverTauPt, float>,
    llama::Field<Tau_mass, float>,
    llama::Field<Tau_neutralIso, float>,
    llama::Field<Tau_phi, float>,
    llama::Field<Tau_photonsOutsideSignalCone, float>,
    llama::Field<Tau_pt, float>,
    llama::Field<Tau_puCorr, float>,
    llama::Field<Tau_rawAntiEle, float>,
    llama::Field<Tau_rawAntiEle2018, float>,
    llama::Field<Tau_rawIso, float>,
    llama::Field<Tau_rawIsodR03, float>,
    llama::Field<Tau_rawMVAnewDM2017v2, float>,
    llama::Field<Tau_rawMVAoldDM, float>,
    llama::Field<Tau_rawMVAoldDM2017v1, float>,
    llama::Field<Tau_rawMVAoldDM2017v2, float>,
    llama::Field<Tau_rawMVAoldDMdR032017v2, float>,
    llama::Field<Tau_charge, std::int32_t>,
    llama::Field<Tau_decayMode, std::int32_t>,
    llama::Field<Tau_jetIdx, std::int32_t>,
    llama::Field<Tau_rawAntiEleCat, std::int32_t>,
    llama::Field<Tau_rawAntiEleCat2018, std::int32_t>,
    llama::Field<Tau_idAntiEle, byte>,
    llama::Field<Tau_idAntiEle2018, byte>,
    llama::Field<Tau_idAntiMu, byte>,
    llama::Field<Tau_idDecayMode, bit>,
    llama::Field<Tau_idDecayModeNewDMs, bit>,
    llama::Field<Tau_idMVAnewDM2017v2, byte>,
    llama::Field<Tau_idMVAoldDM, byte>,
    llama::Field<Tau_idMVAoldDM2017v1, byte>,
    llama::Field<Tau_idMVAoldDM2017v2, byte>,
    llama::Field<Tau_idMVAoldDMdR032017v2, byte>,
    llama::Field<Tau_cleanmask, byte>,
    llama::Field<Tau_genPartIdx, std::int32_t>,
    llama::Field<Tau_genPartFlav, byte>
>;

using TrigObj = llama::Record<
    llama::Field<TrigObj_pt, float>,
    llama::Field<TrigObj_eta, float>,
    llama::Field<TrigObj_phi, float>,
    llama::Field<TrigObj_l1pt, float>,
    llama::Field<TrigObj_l1pt_2, float>,
    llama::Field<TrigObj_l2pt, float>,
    llama::Field<TrigObj_id, std::int32_t>,
    llama::Field<TrigObj_l1iso, std::int32_t>,
    llama::Field<TrigObj_l1charge, std::int32_t>,
    llama::Field<TrigObj_filterbits, std::int32_t>
>;

using OtherPV = llama::Record<
    llama::Field<OtherPV_z, float>
>;

using SV = llama::Record<
    llama::Field<SV_dlen, float>,
    llama::Field<SV_dlenSig, float>,
    llama::Field<SV_pAngle, float>,
    llama::Field<SV_chi2, float>,
    llama::Field<SV_eta, float>,
    llama::Field<SV_mass, float>,
    llama::Field<SV_ndof, float>,
    llama::Field<SV_phi, float>,
    llama::Field<SV_pt, float>,
    llama::Field<SV_x, float>,
    llama::Field<SV_y, float>,
    llama::Field<SV_z, float>
>;

using Event = llama::Record<
    llama::Field<run, std::int32_t>,
    llama::Field<luminosityBlock, std::int32_t>,
    llama::Field<event, std::int64_t>,
    llama::Field<HTXS_Higgs_pt, float>,
    llama::Field<HTXS_Higgs_y, float>,
    llama::Field<HTXS_stage1_1_cat_pTjet25GeV, std::int32_t>,
    llama::Field<HTXS_stage1_1_cat_pTjet30GeV, std::int32_t>,
    llama::Field<HTXS_stage1_1_fine_cat_pTjet25GeV, std::int32_t>,
    llama::Field<HTXS_stage1_1_fine_cat_pTjet30GeV, std::int32_t>,
    llama::Field<HTXS_stage_0, std::int32_t>,
    llama::Field<HTXS_stage_1_pTjet25, std::int32_t>,
    llama::Field<HTXS_stage_1_pTjet30, std::int32_t>,
    llama::Field<HTXS_njets25, byte>,
    llama::Field<HTXS_njets30, byte>,
    llama::Field<btagWeight_CSVV2, float>,
    llama::Field<btagWeight_DeepCSVB, float>,
    llama::Field<CaloMET_phi, float>,
    llama::Field<CaloMET_pt, float>,
    llama::Field<CaloMET_sumEt, float>,
    llama::Field<ChsMET_phi, float>,
    llama::Field<ChsMET_pt, float>,
    llama::Field<ChsMET_sumEt, float>,
    //llama::Field<nCorrT1METJet, Index>,
    //llama::Field<nElectron, Index>,
    llama::Field<nElectron, Electron[]>,
    llama::Field<Flag_ecalBadCalibFilterV2, bit>,
    //llama::Field<nFatJet, Index>,
    //llama::Field<nGenJetAK8, Index>,
    //llama::Field<nGenJet, Index>,
    //llama::Field<nGenPart, Index>,
    //llama::Field<nSubGenJetAK8, Index>,
    llama::Field<Generator_binvar, float>,
    llama::Field<Generator_scalePDF, float>,
    llama::Field<Generator_weight, float>,
    llama::Field<Generator_x1, float>,
    llama::Field<Generator_x2, float>,
    llama::Field<Generator_xpdf1, float>,
    llama::Field<Generator_xpdf2, float>,
    llama::Field<Generator_id1, std::int32_t>,
    llama::Field<Generator_id2, std::int32_t>,
    //llama::Field<nGenVisTau, Index>,
    llama::Field<genWeight, float>,
    llama::Field<LHEWeight_originalXWGTUP, float>,
    //llama::Field<nLHEPdfWeight, Index>,
    //llama::Field<nLHEReweightingWeight, Index>,
    //llama::Field<nLHEScaleWeight, Index>,
    //llama::Field<nPSWeight, Index>,
    //llama::Field<nIsoTrack, Index>,
    //llama::Field<nJet, Index>,
    llama::Field<LHE_HT, float>,
    llama::Field<LHE_HTIncoming, float>,
    llama::Field<LHE_Vpt, float>,
    llama::Field<LHE_Njets, byte>,
    llama::Field<LHE_Nb, byte>,
    llama::Field<LHE_Nc, byte>,
    llama::Field<LHE_Nuds, byte>,
    llama::Field<LHE_Nglu, byte>,
    llama::Field<LHE_NpNLO, byte>,
    llama::Field<LHE_NpLO, byte>,
    //llama::Field<nLHEPart, Index>,
    llama::Field<GenMET_phi, float>,
    llama::Field<GenMET_pt, float>,
    llama::Field<MET_MetUnclustEnUpDeltaX, float>,
    llama::Field<MET_MetUnclustEnUpDeltaY, float>,
    llama::Field<MET_covXX, float>,
    llama::Field<MET_covXY, float>,
    llama::Field<MET_covYY, float>,
    llama::Field<MET_phi, float>,
    llama::Field<MET_pt, float>,
    llama::Field<MET_significance, float>,
    llama::Field<MET_sumEt, float>,
    //llama::Field<nMuon, Index>,
    //llama::Field<nPhoton, Index>,
    llama::Field<Pileup_nTrueInt, float>,
    llama::Field<Pileup_pudensity, float>,
    llama::Field<Pileup_gpudensity, float>,
    llama::Field<Pileup_nPU, std::int32_t>,
    llama::Field<Pileup_sumEOOT, std::int32_t>,
    llama::Field<Pileup_sumLOOT, std::int32_t>,
    llama::Field<PuppiMET_phi, float>,
    llama::Field<PuppiMET_pt, float>,
    llama::Field<PuppiMET_sumEt, float>,
    llama::Field<RawMET_phi, float>,
    llama::Field<RawMET_pt, float>,
    llama::Field<RawMET_sumEt, float>,
    llama::Field<fixedGridRhoFastjetAll, float>,
    llama::Field<fixedGridRhoFastjetCentral, float>,
    llama::Field<fixedGridRhoFastjetCentralCalo, float>,
    llama::Field<fixedGridRhoFastjetCentralChargedPileUp, float>,
    llama::Field<fixedGridRhoFastjetCentralNeutral, float>,
    //llama::Field<nGenDressedLepton, Index>,
    //llama::Field<nSoftActivityJet, Index>,
    llama::Field<SoftActivityJetHT, float>,
    llama::Field<SoftActivityJetHT10, float>,
    llama::Field<SoftActivityJetHT2, float>,
    llama::Field<SoftActivityJetHT5, float>,
    llama::Field<SoftActivityJetNjets10, std::int32_t>,
    llama::Field<SoftActivityJetNjets2, std::int32_t>,
    llama::Field<SoftActivityJetNjets5, std::int32_t>,
    //llama::Field<nSubJet, Index>,
    //llama::Field<nTau, Index>,
    llama::Field<TkMET_phi, float>,
    llama::Field<TkMET_pt, float>,
    llama::Field<TkMET_sumEt, float>,
    //llama::Field<nTrigObj, Index>,
    llama::Field<genTtbarId, std::int32_t>,
    //llama::Field<nOtherPV, Index>,
    llama::Field<PV_ndof, float>,
    llama::Field<PV_x, float>,
    llama::Field<PV_y, float>,
    llama::Field<PV_z, float>,
    llama::Field<PV_chi2, float>,
    llama::Field<PV_score, float>,
    llama::Field<PV_npvs, std::int32_t>,
    llama::Field<PV_npvsGood, std::int32_t>,
    //llama::Field<nSV, Index>,
    llama::Field<MET_fiducialGenPhi, float>,
    llama::Field<MET_fiducialGenPt, float>,
    llama::Field<L1simulation_step, bit>,
    llama::Field<HLTriggerFirstPath, bit>,
    llama::Field<HLT_AK8PFJet360_TrimMass30, bit>,
    llama::Field<HLT_AK8PFJet380_TrimMass30, bit>,
    llama::Field<HLT_AK8PFJet400_TrimMass30, bit>,
    llama::Field<HLT_AK8PFJet420_TrimMass30, bit>,
    llama::Field<HLT_AK8PFHT750_TrimMass50, bit>,
    llama::Field<HLT_AK8PFHT800_TrimMass50, bit>,
    llama::Field<HLT_AK8PFHT850_TrimMass50, bit>,
    llama::Field<HLT_AK8PFHT900_TrimMass50, bit>,
    llama::Field<HLT_CaloJet500_NoJetID, bit>,
    llama::Field<HLT_CaloJet550_NoJetID, bit>,
    llama::Field<HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL, bit>,
    llama::Field<HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon, bit>,
    llama::Field<HLT_Trimuon5_3p5_2_Upsilon_Muon, bit>,
    llama::Field<HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon, bit>,
    llama::Field<HLT_DoubleEle25_CaloIdL_MW, bit>,
    llama::Field<HLT_DoubleEle27_CaloIdL_MW, bit>,
    llama::Field<HLT_DoubleEle33_CaloIdL_MW, bit>,
    llama::Field<HLT_DoubleEle24_eta2p1_WPTight_Gsf, bit>,
    llama::Field<HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350, bit>,
    llama::Field<HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350, bit>,
    llama::Field<HLT_Ele27_Ele37_CaloIdL_MW, bit>,
    llama::Field<HLT_Mu27_Ele37_CaloIdL_MW, bit>,
    llama::Field<HLT_Mu37_Ele27_CaloIdL_MW, bit>,
    llama::Field<HLT_Mu37_TkMu27, bit>,
    llama::Field<HLT_DoubleMu4_3_Bs, bit>,
    llama::Field<HLT_DoubleMu4_3_Jpsi, bit>,
    llama::Field<HLT_DoubleMu4_JpsiTrk_Displaced, bit>,
    llama::Field<HLT_DoubleMu4_LowMassNonResonantTrk_Displaced, bit>,
    llama::Field<HLT_DoubleMu3_Trk_Tau3mu, bit>,
    llama::Field<HLT_DoubleMu3_TkMu_DsTau3Mu, bit>,
    llama::Field<HLT_DoubleMu4_PsiPrimeTrk_Displaced, bit>,
    llama::Field<HLT_DoubleMu4_Mass3p8_DZ_PFHT350, bit>,
    llama::Field<HLT_Mu3_PFJet40, bit>,
    llama::Field<HLT_Mu7p5_L2Mu2_Jpsi, bit>,
    llama::Field<HLT_Mu7p5_L2Mu2_Upsilon, bit>,
    llama::Field<HLT_Mu7p5_Track2_Jpsi, bit>,
    llama::Field<HLT_Mu7p5_Track3p5_Jpsi, bit>,
    llama::Field<HLT_Mu7p5_Track7_Jpsi, bit>,
    llama::Field<HLT_Mu7p5_Track2_Upsilon, bit>,
    llama::Field<HLT_Mu7p5_Track3p5_Upsilon, bit>,
    llama::Field<HLT_Mu7p5_Track7_Upsilon, bit>,
    llama::Field<HLT_Mu3_L1SingleMu5orSingleMu7, bit>,
    llama::Field<HLT_DoublePhoton33_CaloIdL, bit>,
    llama::Field<HLT_DoublePhoton70, bit>,
    llama::Field<HLT_DoublePhoton85, bit>,
    llama::Field<HLT_Ele20_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele15_WPLoose_Gsf, bit>,
    llama::Field<HLT_Ele17_WPLoose_Gsf, bit>,
    llama::Field<HLT_Ele20_WPLoose_Gsf, bit>,
    llama::Field<HLT_Ele20_eta2p1_WPLoose_Gsf, bit>,
    llama::Field<HLT_DiEle27_WPTightCaloOnly_L1DoubleEG, bit>,
    llama::Field<HLT_Ele27_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele28_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele30_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele32_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele35_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele35_WPTight_Gsf_L1EGMT, bit>,
    llama::Field<HLT_Ele38_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele40_WPTight_Gsf, bit>,
    llama::Field<HLT_Ele32_WPTight_Gsf_L1DoubleEG, bit>,
    llama::Field<HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1, bit>,
    llama::Field<HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1, bit>,
    llama::Field<HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1, bit>,
    llama::Field<HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1, bit>,
    llama::Field<HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1, bit>,
    llama::Field<HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_TightID_CrossL1, bit>,
    llama::Field<HLT_HT450_Beamspot, bit>,
    llama::Field<HLT_HT300_Beamspot, bit>,
    llama::Field<HLT_ZeroBias_Beamspot, bit>,
    llama::Field<HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1, bit>,
    llama::Field<HLT_IsoMu20_eta2p1_MediumChargedIsoPFTauHPS27_eta2p1_CrossL1, bit>,
    llama::Field<HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_CrossL1, bit>,
    llama::Field<HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1, bit>,
    llama::Field<HLT_IsoMu20_eta2p1_MediumChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1, bit>,
    llama::Field<HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1, bit>,
    llama::Field<HLT_IsoMu24_eta2p1_TightChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_CrossL1, bit>,
    llama::Field<HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_CrossL1, bit>,
    llama::Field<HLT_IsoMu24_eta2p1_TightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg_CrossL1, bit>,
    llama::Field<HLT_IsoMu24_eta2p1_MediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_CrossL1, bit>,
    llama::Field<HLT_IsoMu27_LooseChargedIsoPFTauHPS20_Trk1_eta2p1_SingleL1, bit>,
    llama::Field<HLT_IsoMu27_MediumChargedIsoPFTauHPS20_Trk1_eta2p1_SingleL1, bit>,
    llama::Field<HLT_IsoMu27_TightChargedIsoPFTauHPS20_Trk1_eta2p1_SingleL1, bit>,
    llama::Field<HLT_IsoMu20, bit>,
    llama::Field<HLT_IsoMu24, bit>,
    llama::Field<HLT_IsoMu24_eta2p1, bit>,
    llama::Field<HLT_IsoMu27, bit>,
    llama::Field<HLT_IsoMu30, bit>,
    llama::Field<HLT_UncorrectedJetE30_NoBPTX, bit>,
    llama::Field<HLT_UncorrectedJetE30_NoBPTX3BX, bit>,
    llama::Field<HLT_UncorrectedJetE60_NoBPTX3BX, bit>,
    llama::Field<HLT_UncorrectedJetE70_NoBPTX3BX, bit>,
    llama::Field<HLT_L1SingleMu18, bit>,
    llama::Field<HLT_L1SingleMu25, bit>,
    llama::Field<HLT_L2Mu10, bit>,
    llama::Field<HLT_L2Mu10_NoVertex_NoBPTX3BX, bit>,
    llama::Field<HLT_L2Mu10_NoVertex_NoBPTX, bit>,
    llama::Field<HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX, bit>,
    llama::Field<HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX, bit>,
    llama::Field<HLT_L2Mu50, bit>,
    llama::Field<HLT_L2Mu23NoVtx_2Cha, bit>,
    llama::Field<HLT_L2Mu23NoVtx_2Cha_CosmicSeed, bit>,
    llama::Field<HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4, bit>,
    llama::Field<HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4, bit>,
    llama::Field<HLT_DoubleL2Mu50, bit>,
    llama::Field<HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed, bit>,
    llama::Field<HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_NoL2Matched, bit>,
    llama::Field<HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed, bit>,
    llama::Field<HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_NoL2Matched, bit>,
    llama::Field<HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4, bit>,
    llama::Field<HLT_DoubleL2Mu23NoVtx_2Cha, bit>,
    llama::Field<HLT_DoubleL2Mu23NoVtx_2Cha_NoL2Matched, bit>,
    llama::Field<HLT_DoubleL2Mu25NoVtx_2Cha, bit>,
    llama::Field<HLT_DoubleL2Mu25NoVtx_2Cha_NoL2Matched, bit>,
    llama::Field<HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4, bit>,
    llama::Field<HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL, bit>,
    llama::Field<HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL, bit>,
    llama::Field<HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ, bit>,
    llama::Field<HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ, bit>,
    llama::Field<HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8, bit>,
    llama::Field<HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8, bit>,
    llama::Field<HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, bit>,
    llama::Field<HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8, bit>,
    llama::Field<HLT_Mu25_TkMu0_Onia, bit>,
    llama::Field<HLT_Mu30_TkMu0_Psi, bit>,
    llama::Field<HLT_Mu30_TkMu0_Upsilon, bit>,
    llama::Field<HLT_Mu20_TkMu0_Phi, bit>,
    llama::Field<HLT_Mu25_TkMu0_Phi, bit>,
    llama::Field<HLT_Mu12, bit>,
    llama::Field<HLT_Mu15, bit>,
    llama::Field<HLT_Mu20, bit>,
    llama::Field<HLT_Mu27, bit>,
    llama::Field<HLT_Mu50, bit>,
    llama::Field<HLT_Mu55, bit>,
    llama::Field<HLT_OldMu100, bit>,
    llama::Field<HLT_TkMu100, bit>,
    llama::Field<HLT_DiPFJetAve40, bit>,
    llama::Field<HLT_DiPFJetAve60, bit>,
    llama::Field<HLT_DiPFJetAve80, bit>,
    llama::Field<HLT_DiPFJetAve140, bit>,
    llama::Field<HLT_DiPFJetAve200, bit>,
    llama::Field<HLT_DiPFJetAve260, bit>,
    llama::Field<HLT_DiPFJetAve320, bit>,
    llama::Field<HLT_DiPFJetAve400, bit>,
    llama::Field<HLT_DiPFJetAve500, bit>,
    llama::Field<HLT_DiPFJetAve60_HFJEC, bit>,
    llama::Field<HLT_DiPFJetAve80_HFJEC, bit>,
    llama::Field<HLT_DiPFJetAve100_HFJEC, bit>,
    llama::Field<HLT_DiPFJetAve160_HFJEC, bit>,
    llama::Field<HLT_DiPFJetAve220_HFJEC, bit>,
    llama::Field<HLT_DiPFJetAve300_HFJEC, bit>,
    llama::Field<HLT_AK8PFJet15, bit>,
    llama::Field<HLT_AK8PFJet25, bit>,
    llama::Field<HLT_AK8PFJet40, bit>,
    llama::Field<HLT_AK8PFJet60, bit>,
    llama::Field<HLT_AK8PFJet80, bit>,
    llama::Field<HLT_AK8PFJet140, bit>,
    llama::Field<HLT_AK8PFJet200, bit>,
    llama::Field<HLT_AK8PFJet260, bit>,
    llama::Field<HLT_AK8PFJet320, bit>,
    llama::Field<HLT_AK8PFJet400, bit>,
    llama::Field<HLT_AK8PFJet450, bit>,
    llama::Field<HLT_AK8PFJet500, bit>,
    llama::Field<HLT_AK8PFJet550, bit>,
    llama::Field<HLT_PFJet15, bit>,
    llama::Field<HLT_PFJet25, bit>,
    llama::Field<HLT_PFJet40, bit>,
    llama::Field<HLT_PFJet60, bit>,
    llama::Field<HLT_PFJet80, bit>,
    llama::Field<HLT_PFJet140, bit>,
    llama::Field<HLT_PFJet200, bit>,
    llama::Field<HLT_PFJet260, bit>,
    llama::Field<HLT_PFJet320, bit>,
    llama::Field<HLT_PFJet400, bit>,
    llama::Field<HLT_PFJet450, bit>,
    llama::Field<HLT_PFJet500, bit>,
    llama::Field<HLT_PFJet550, bit>,
    llama::Field<HLT_PFJetFwd15, bit>,
    llama::Field<HLT_PFJetFwd25, bit>,
    llama::Field<HLT_PFJetFwd40, bit>,
    llama::Field<HLT_PFJetFwd60, bit>,
    llama::Field<HLT_PFJetFwd80, bit>,
    llama::Field<HLT_PFJetFwd140, bit>,
    llama::Field<HLT_PFJetFwd200, bit>,
    llama::Field<HLT_PFJetFwd260, bit>,
    llama::Field<HLT_PFJetFwd320, bit>,
    llama::Field<HLT_PFJetFwd400, bit>,
    llama::Field<HLT_PFJetFwd450, bit>,
    llama::Field<HLT_PFJetFwd500, bit>,
    llama::Field<HLT_AK8PFJetFwd15, bit>,
    llama::Field<HLT_AK8PFJetFwd25, bit>,
    llama::Field<HLT_AK8PFJetFwd40, bit>,
    llama::Field<HLT_AK8PFJetFwd60, bit>,
    llama::Field<HLT_AK8PFJetFwd80, bit>,
    llama::Field<HLT_AK8PFJetFwd140, bit>,
    llama::Field<HLT_AK8PFJetFwd200, bit>,
    llama::Field<HLT_AK8PFJetFwd260, bit>,
    llama::Field<HLT_AK8PFJetFwd320, bit>,
    llama::Field<HLT_AK8PFJetFwd400, bit>,
    llama::Field<HLT_AK8PFJetFwd450, bit>,
    llama::Field<HLT_AK8PFJetFwd500, bit>,
    llama::Field<HLT_PFHT180, bit>,
    llama::Field<HLT_PFHT250, bit>,
    llama::Field<HLT_PFHT370, bit>,
    llama::Field<HLT_PFHT430, bit>,
    llama::Field<HLT_PFHT510, bit>,
    llama::Field<HLT_PFHT590, bit>,
    llama::Field<HLT_PFHT680, bit>,
    llama::Field<HLT_PFHT780, bit>,
    llama::Field<HLT_PFHT890, bit>,
    llama::Field<HLT_PFHT1050, bit>,
    llama::Field<HLT_PFHT500_PFMET100_PFMHT100_IDTight, bit>,
    llama::Field<HLT_PFHT500_PFMET110_PFMHT110_IDTight, bit>,
    llama::Field<HLT_PFHT700_PFMET85_PFMHT85_IDTight, bit>,
    llama::Field<HLT_PFHT700_PFMET95_PFMHT95_IDTight, bit>,
    llama::Field<HLT_PFHT800_PFMET75_PFMHT75_IDTight, bit>,
    llama::Field<HLT_PFHT800_PFMET85_PFMHT85_IDTight, bit>,
    llama::Field<HLT_PFMET110_PFMHT110_IDTight, bit>,
    llama::Field<HLT_PFMET120_PFMHT120_IDTight, bit>,
    llama::Field<HLT_PFMET130_PFMHT130_IDTight, bit>,
    llama::Field<HLT_PFMET140_PFMHT140_IDTight, bit>,
    llama::Field<HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1, bit>,
    llama::Field<HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1, bit>,
    llama::Field<HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1, bit>,
    llama::Field<HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1, bit>,
    llama::Field<HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1, bit>,
    llama::Field<HLT_PFMET120_PFMHT120_IDTight_PFHT60, bit>,
    llama::Field<HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60, bit>,
    llama::Field<HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60, bit>,
    llama::Field<HLT_PFMETTypeOne110_PFMHT110_IDTight, bit>,
    llama::Field<HLT_PFMETTypeOne120_PFMHT120_IDTight, bit>,
    llama::Field<HLT_PFMETTypeOne130_PFMHT130_IDTight, bit>,
    llama::Field<HLT_PFMETTypeOne140_PFMHT140_IDTight, bit>,
    llama::Field<HLT_PFMETNoMu110_PFMHTNoMu110_IDTight, bit>,
    llama::Field<HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, bit>,
    llama::Field<HLT_PFMETNoMu130_PFMHTNoMu130_IDTight, bit>,
    llama::Field<HLT_PFMETNoMu140_PFMHTNoMu140_IDTight, bit>,
    llama::Field<HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight, bit>,
    llama::Field<HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight, bit>,
    llama::Field<HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight, bit>,
    llama::Field<HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight, bit>,
    llama::Field<HLT_L1ETMHadSeeds, bit>,
    llama::Field<HLT_CaloMHT90, bit>,
    llama::Field<HLT_CaloMET80_NotCleaned, bit>,
    llama::Field<HLT_CaloMET90_NotCleaned, bit>,
    llama::Field<HLT_CaloMET100_NotCleaned, bit>,
    llama::Field<HLT_CaloMET110_NotCleaned, bit>,
    llama::Field<HLT_CaloMET250_NotCleaned, bit>,
    llama::Field<HLT_CaloMET70_HBHECleaned, bit>,
    llama::Field<HLT_CaloMET80_HBHECleaned, bit>,
    llama::Field<HLT_CaloMET90_HBHECleaned, bit>,
    llama::Field<HLT_CaloMET100_HBHECleaned, bit>,
    llama::Field<HLT_CaloMET250_HBHECleaned, bit>,
    llama::Field<HLT_CaloMET300_HBHECleaned, bit>,
    llama::Field<HLT_CaloMET350_HBHECleaned, bit>,
    llama::Field<HLT_PFMET200_NotCleaned, bit>,
    llama::Field<HLT_PFMET200_HBHECleaned, bit>,
    llama::Field<HLT_PFMET250_HBHECleaned, bit>,
    llama::Field<HLT_PFMET300_HBHECleaned, bit>,
    llama::Field<HLT_PFMET200_HBHE_BeamHaloCleaned, bit>,
    llama::Field<HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned, bit>,
    llama::Field<HLT_MET105_IsoTrk50, bit>,
    llama::Field<HLT_MET120_IsoTrk50, bit>,
    llama::Field<HLT_SingleJet30_Mu12_SinglePFJet40, bit>,
    llama::Field<HLT_Mu12_DoublePFJets40_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Mu12_DoublePFJets100_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Mu12_DoublePFJets200_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Mu12_DoublePFJets350_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Mu12_DoublePFJets54MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Mu12_DoublePFJets62MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_DoublePFJets40_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_DoublePFJets100_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_DoublePFJets200_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_DoublePFJets350_CaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71, bit>,
    llama::Field<HLT_Photon300_NoHE, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL, bit>,
    llama::Field<HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ, bit>,
    llama::Field<HLT_Mu8_DiEle12_CaloIdL_TrackIdL, bit>,
    llama::Field<HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ, bit>,
    llama::Field<HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL, bit>,
    llama::Field<HLT_Mu17_TrkIsoVVL, bit>,
    llama::Field<HLT_Mu19_TrkIsoVVL, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet20_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet40_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet70_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet110_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet170_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK4Jet300_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK8DiJet170_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK8Jet170_DoubleMu5, bit>,
    llama::Field<HLT_BTagMu_AK8Jet300_Mu5, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet20_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet40_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet70_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet110_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK4DiJet170_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK4Jet300_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK8DiJet170_Mu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK8Jet170_DoubleMu5_noalgo, bit>,
    llama::Field<HLT_BTagMu_AK8Jet300_Mu5_noalgo, bit>,
    llama::Field<HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL, bit>,
    llama::Field<HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, bit>,
    llama::Field<HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL, bit>,
    llama::Field<HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, bit>,
    llama::Field<HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL, bit>,
    llama::Field<HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL, bit>,
    llama::Field<HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ, bit>,
    llama::Field<HLT_Mu12_DoublePhoton20, bit>,
    llama::Field<HLT_TriplePhoton_20_20_20_CaloIdLV2, bit>,
    llama::Field<HLT_TriplePhoton_20_20_20_CaloIdLV2_R9IdVL, bit>,
    llama::Field<HLT_TriplePhoton_30_30_10_CaloIdLV2, bit>,
    llama::Field<HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL, bit>,
    llama::Field<HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL, bit>,
    llama::Field<HLT_Photon20, bit>,
    llama::Field<HLT_Photon33, bit>,
    llama::Field<HLT_Photon50, bit>,
    llama::Field<HLT_Photon75, bit>,
    llama::Field<HLT_Photon90, bit>,
    llama::Field<HLT_Photon120, bit>,
    llama::Field<HLT_Photon150, bit>,
    llama::Field<HLT_Photon175, bit>,
    llama::Field<HLT_Photon200, bit>,
    llama::Field<HLT_Photon100EB_TightID_TightIso, bit>,
    llama::Field<HLT_Photon110EB_TightID_TightIso, bit>,
    llama::Field<HLT_Photon120EB_TightID_TightIso, bit>,
    llama::Field<HLT_Photon100EBHE10, bit>,
    llama::Field<HLT_Photon100EEHE10, bit>,
    llama::Field<HLT_Photon100EE_TightID_TightIso, bit>,
    llama::Field<HLT_Photon50_R9Id90_HE10_IsoM, bit>,
    llama::Field<HLT_Photon75_R9Id90_HE10_IsoM, bit>,
    llama::Field<HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3, bit>,
    llama::Field<HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3, bit>,
    llama::Field<HLT_Photon90_R9Id90_HE10_IsoM, bit>,
    llama::Field<HLT_Photon120_R9Id90_HE10_IsoM, bit>,
    llama::Field<HLT_Photon165_R9Id90_HE10_IsoM, bit>,
    llama::Field<HLT_Photon90_CaloIdL_PFHT700, bit>,
    llama::Field<HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90, bit>,
    llama::Field<HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95, bit>,
    llama::Field<HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55, bit>,
    llama::Field<HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55, bit>,
    llama::Field<HLT_Photon35_TwoProngs35, bit>,
    llama::Field<HLT_IsoMu24_TwoProngs35, bit>,
    llama::Field<HLT_Dimuon0_Jpsi_L1_NoOS, bit>,
    llama::Field<HLT_Dimuon0_Jpsi_NoVertexing_NoOS, bit>,
    llama::Field<HLT_Dimuon0_Jpsi, bit>,
    llama::Field<HLT_Dimuon0_Jpsi_NoVertexing, bit>,
    llama::Field<HLT_Dimuon0_Jpsi_L1_4R_0er1p5R, bit>,
    llama::Field<HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R, bit>,
    llama::Field<HLT_Dimuon0_Jpsi3p5_Muon2, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_L1_4p5, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_L1_5, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_L1_4p5NoOS, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_L1_4p5er2p0, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_L1_4p5er2p0M, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_NoVertexing, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_L1_5M, bit>,
    llama::Field<HLT_Dimuon0_LowMass_L1_0er1p5R, bit>,
    llama::Field<HLT_Dimuon0_LowMass_L1_0er1p5, bit>,
    llama::Field<HLT_Dimuon0_LowMass, bit>,
    llama::Field<HLT_Dimuon0_LowMass_L1_4, bit>,
    llama::Field<HLT_Dimuon0_LowMass_L1_4R, bit>,
    llama::Field<HLT_Dimuon0_LowMass_L1_TM530, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_Muon_L1_TM0, bit>,
    llama::Field<HLT_Dimuon0_Upsilon_Muon_NoL1Mass, bit>,
    llama::Field<HLT_TripleMu_5_3_3_Mass3p8_DZ, bit>,
    llama::Field<HLT_TripleMu_10_5_5_DZ, bit>,
    llama::Field<HLT_TripleMu_12_10_5, bit>,
    llama::Field<HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15, bit>,
    llama::Field<HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1, bit>,
    llama::Field<HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15, bit>,
    llama::Field<HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1, bit>,
    llama::Field<HLT_DoubleMu3_DZ_PFMET50_PFMHT60, bit>,
    llama::Field<HLT_DoubleMu3_DZ_PFMET70_PFMHT70, bit>,
    llama::Field<HLT_DoubleMu3_DZ_PFMET90_PFMHT90, bit>,
    llama::Field<HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass, bit>,
    llama::Field<HLT_DoubleMu4_Jpsi_Displaced, bit>,
    llama::Field<HLT_DoubleMu4_Jpsi_NoVertexing, bit>,
    llama::Field<HLT_DoubleMu4_JpsiTrkTrk_Displaced, bit>,
    llama::Field<HLT_DoubleMu43NoFiltersNoVtx, bit>,
    llama::Field<HLT_DoubleMu48NoFiltersNoVtx, bit>,
    llama::Field<HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL, bit>,
    llama::Field<HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL, bit>,
    llama::Field<HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL, bit>,
    llama::Field<HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL, bit>,
    llama::Field<HLT_DoubleMu33NoFiltersNoVtxDisplaced, bit>,
    llama::Field<HLT_DoubleMu40NoFiltersNoVtxDisplaced, bit>,
    llama::Field<HLT_DoubleMu20_7_Mass0to30_L1_DM4, bit>,
    llama::Field<HLT_DoubleMu20_7_Mass0to30_L1_DM4EG, bit>,
    llama::Field<HLT_HT425, bit>,
    llama::Field<HLT_HT430_DisplacedDijet40_DisplacedTrack, bit>,
    llama::Field<HLT_HT500_DisplacedDijet40_DisplacedTrack, bit>,
    llama::Field<HLT_HT430_DisplacedDijet60_DisplacedTrack, bit>,
    llama::Field<HLT_HT400_DisplacedDijet40_DisplacedTrack, bit>,
    llama::Field<HLT_HT650_DisplacedDijet60_Inclusive, bit>,
    llama::Field<HLT_HT550_DisplacedDijet60_Inclusive, bit>,
    llama::Field<HLT_DiJet110_35_Mjj650_PFMET110, bit>,
    llama::Field<HLT_DiJet110_35_Mjj650_PFMET120, bit>,
    llama::Field<HLT_DiJet110_35_Mjj650_PFMET130, bit>,
    llama::Field<HLT_TripleJet110_35_35_Mjj650_PFMET110, bit>,
    llama::Field<HLT_TripleJet110_35_35_Mjj650_PFMET120, bit>,
    llama::Field<HLT_TripleJet110_35_35_Mjj650_PFMET130, bit>,
    llama::Field<HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned, bit>,
    llama::Field<HLT_Ele28_eta2p1_WPTight_Gsf_HT150, bit>,
    llama::Field<HLT_Ele28_HighEta_SC20_Mass55, bit>,
    llama::Field<HLT_DoubleMu20_7_Mass0to30_Photon23, bit>,
    llama::Field<HLT_Ele15_IsoVVVL_PFHT450_CaloBTagDeepCSV_4p5, bit>,
    llama::Field<HLT_Ele15_IsoVVVL_PFHT450_PFMET50, bit>,
    llama::Field<HLT_Ele15_IsoVVVL_PFHT450, bit>,
    llama::Field<HLT_Ele50_IsoVVVL_PFHT450, bit>,
    llama::Field<HLT_Ele15_IsoVVVL_PFHT600, bit>,
    llama::Field<HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60, bit>,
    llama::Field<HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60, bit>,
    llama::Field<HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60, bit>,
    llama::Field<HLT_Mu15_IsoVVVL_PFHT450_CaloBTagDeepCSV_4p5, bit>,
    llama::Field<HLT_Mu15_IsoVVVL_PFHT450_PFMET50, bit>,
    llama::Field<HLT_Mu15_IsoVVVL_PFHT450, bit>,
    llama::Field<HLT_Mu50_IsoVVVL_PFHT450, bit>,
    llama::Field<HLT_Mu15_IsoVVVL_PFHT600, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMET70_PFMHT70_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu70_PFMHTNoMu70_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight, bit>,
    llama::Field<HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight, bit>,
    llama::Field<HLT_Dimuon10_PsiPrime_Barrel_Seagulls, bit>,
    llama::Field<HLT_Dimuon20_Jpsi_Barrel_Seagulls, bit>,
    llama::Field<HLT_Dimuon12_Upsilon_y1p4, bit>,
    llama::Field<HLT_Dimuon14_Phi_Barrel_Seagulls, bit>,
    llama::Field<HLT_Dimuon18_PsiPrime, bit>,
    llama::Field<HLT_Dimuon25_Jpsi, bit>,
    llama::Field<HLT_Dimuon18_PsiPrime_noCorrL1, bit>,
    llama::Field<HLT_Dimuon24_Upsilon_noCorrL1, bit>,
    llama::Field<HLT_Dimuon24_Phi_noCorrL1, bit>,
    llama::Field<HLT_Dimuon25_Jpsi_noCorrL1, bit>,
    llama::Field<HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8, bit>,
    llama::Field<HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ, bit>,
    llama::Field<HLT_DiMu9_Ele9_CaloIdL_TrackIdL, bit>,
    llama::Field<HLT_DoubleIsoMu20_eta2p1, bit>,
    llama::Field<HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx, bit>,
    llama::Field<HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx, bit>,
    llama::Field<HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx, bit>,
    llama::Field<HLT_Mu8, bit>,
    llama::Field<HLT_Mu17, bit>,
    llama::Field<HLT_Mu19, bit>,
    llama::Field<HLT_Mu17_Photon30_IsoCaloId, bit>,
    llama::Field<HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30, bit>,
    llama::Field<HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30, bit>,
    llama::Field<HLT_Ele15_CaloIdL_TrackIdL_IsoVL_PFJet30, bit>,
    llama::Field<HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30, bit>,
    llama::Field<HLT_Ele8_CaloIdM_TrackIdM_PFJet30, bit>,
    llama::Field<HLT_Ele17_CaloIdM_TrackIdM_PFJet30, bit>,
    llama::Field<HLT_Ele23_CaloIdM_TrackIdM_PFJet30, bit>,
    llama::Field<HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165, bit>,
    llama::Field<HLT_Ele115_CaloIdVT_GsfTrkIdT, bit>,
    llama::Field<HLT_Ele135_CaloIdVT_GsfTrkIdT, bit>,
    llama::Field<HLT_Ele145_CaloIdVT_GsfTrkIdT, bit>,
    llama::Field<HLT_Ele200_CaloIdVT_GsfTrkIdT, bit>,
    llama::Field<HLT_Ele250_CaloIdVT_GsfTrkIdT, bit>,
    llama::Field<HLT_Ele300_CaloIdVT_GsfTrkIdT, bit>,
    llama::Field<HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5, bit>,
    llama::Field<HLT_PFHT330PT30_QuadPFJet_75_60_45_40, bit>,
    llama::Field<HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94, bit>,
    llama::Field<HLT_PFHT400_SixPFJet32, bit>,
    llama::Field<HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59, bit>,
    llama::Field<HLT_PFHT450_SixPFJet36, bit>,
    llama::Field<HLT_PFHT350, bit>,
    llama::Field<HLT_PFHT350MinPFJet15, bit>,
    llama::Field<HLT_Photon60_R9Id90_CaloIdL_IsoL, bit>,
    llama::Field<HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL, bit>,
    llama::Field<HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15, bit>,
    llama::Field<HLT_ECALHT800, bit>,
    llama::Field<HLT_DiSC30_18_EIso_AND_HE_Mass70, bit>,
    llama::Field<HLT_Physics, bit>,
    llama::Field<HLT_Physics_part0, bit>,
    llama::Field<HLT_Physics_part1, bit>,
    llama::Field<HLT_Physics_part2, bit>,
    llama::Field<HLT_Physics_part3, bit>,
    llama::Field<HLT_Physics_part4, bit>,
    llama::Field<HLT_Physics_part5, bit>,
    llama::Field<HLT_Physics_part6, bit>,
    llama::Field<HLT_Physics_part7, bit>,
    llama::Field<HLT_Random, bit>,
    llama::Field<HLT_ZeroBias, bit>,
    llama::Field<HLT_ZeroBias_Alignment, bit>,
    llama::Field<HLT_ZeroBias_part0, bit>,
    llama::Field<HLT_ZeroBias_part1, bit>,
    llama::Field<HLT_ZeroBias_part2, bit>,
    llama::Field<HLT_ZeroBias_part3, bit>,
    llama::Field<HLT_ZeroBias_part4, bit>,
    llama::Field<HLT_ZeroBias_part5, bit>,
    llama::Field<HLT_ZeroBias_part6, bit>,
    llama::Field<HLT_ZeroBias_part7, bit>,
    llama::Field<HLT_AK4CaloJet30, bit>,
    llama::Field<HLT_AK4CaloJet40, bit>,
    llama::Field<HLT_AK4CaloJet50, bit>,
    llama::Field<HLT_AK4CaloJet80, bit>,
    llama::Field<HLT_AK4CaloJet100, bit>,
    llama::Field<HLT_AK4CaloJet120, bit>,
    llama::Field<HLT_AK4PFJet30, bit>,
    llama::Field<HLT_AK4PFJet50, bit>,
    llama::Field<HLT_AK4PFJet80, bit>,
    llama::Field<HLT_AK4PFJet100, bit>,
    llama::Field<HLT_AK4PFJet120, bit>,
    llama::Field<HLT_SinglePhoton10_Eta3p1ForPPRef, bit>,
    llama::Field<HLT_SinglePhoton20_Eta3p1ForPPRef, bit>,
    llama::Field<HLT_SinglePhoton30_Eta3p1ForPPRef, bit>,
    llama::Field<HLT_Photon20_HoverELoose, bit>,
    llama::Field<HLT_Photon30_HoverELoose, bit>,
    llama::Field<HLT_EcalCalibration, bit>,
    llama::Field<HLT_HcalCalibration, bit>,
    llama::Field<HLT_L1UnpairedBunchBptxMinus, bit>,
    llama::Field<HLT_L1UnpairedBunchBptxPlus, bit>,
    llama::Field<HLT_L1NotBptxOR, bit>,
    llama::Field<HLT_L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142, bit>,
    llama::Field<HLT_CDC_L2cosmic_5_er1p0, bit>,
    llama::Field<HLT_CDC_L2cosmic_5p5_er1p0, bit>,
    llama::Field<HLT_HcalNZS, bit>,
    llama::Field<HLT_HcalPhiSym, bit>,
    llama::Field<HLT_HcalIsolatedbunch, bit>,
    llama::Field<HLT_IsoTrackHB, bit>,
    llama::Field<HLT_IsoTrackHE, bit>,
    llama::Field<HLT_ZeroBias_FirstCollisionAfterAbortGap, bit>,
    llama::Field<HLT_ZeroBias_IsolatedBunches, bit>,
    llama::Field<HLT_ZeroBias_FirstCollisionInTrain, bit>,
    llama::Field<HLT_ZeroBias_LastCollisionInTrain, bit>,
    llama::Field<HLT_ZeroBias_FirstBXAfterTrain, bit>,
    llama::Field<HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET110, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET120, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET130, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET140, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau200HighPtRelaxedIso_Trk50_eta2p1, bit>,
    llama::Field<HLT_MediumChargedIsoPFTau220HighPtRelaxedIso_Trk50_eta2p1, bit>,
    llama::Field<HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL, bit>,
    llama::Field<HLT_Rsq0p35, bit>,
    llama::Field<HLT_Rsq0p40, bit>,
    llama::Field<HLT_RsqMR300_Rsq0p09_MR200, bit>,
    llama::Field<HLT_RsqMR320_Rsq0p09_MR200, bit>,
    llama::Field<HLT_RsqMR300_Rsq0p09_MR200_4jet, bit>,
    llama::Field<HLT_RsqMR320_Rsq0p09_MR200_4jet, bit>,
    llama::Field<HLT_IsoMu27_MET90, bit>,
    llama::Field<HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleTightChargedIsoPFTauHPS40_Trk1_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg, bit>,
    llama::Field<HLT_DoubleTightChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg, bit>,
    llama::Field<HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1, bit>,
    llama::Field<HLT_VBF_DoubleMediumChargedIsoPFTauHPS20_Trk1_eta2p1, bit>,
    llama::Field<HLT_VBF_DoubleTightChargedIsoPFTauHPS20_Trk1_eta2p1, bit>,
    llama::Field<HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50, bit>,
    llama::Field<HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3, bit>,
    llama::Field<HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3, bit>,
    llama::Field<HLT_PFMET100_PFMHT100_IDTight_PFHT60, bit>,
    llama::Field<HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60, bit>,
    llama::Field<HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60, bit>,
    llama::Field<HLT_Mu18_Mu9_SameSign, bit>,
    llama::Field<HLT_Mu18_Mu9_SameSign_DZ, bit>,
    llama::Field<HLT_Mu18_Mu9, bit>,
    llama::Field<HLT_Mu18_Mu9_DZ, bit>,
    llama::Field<HLT_Mu20_Mu10_SameSign, bit>,
    llama::Field<HLT_Mu20_Mu10_SameSign_DZ, bit>,
    llama::Field<HLT_Mu20_Mu10, bit>,
    llama::Field<HLT_Mu20_Mu10_DZ, bit>,
    llama::Field<HLT_Mu23_Mu12_SameSign, bit>,
    llama::Field<HLT_Mu23_Mu12_SameSign_DZ, bit>,
    llama::Field<HLT_Mu23_Mu12, bit>,
    llama::Field<HLT_Mu23_Mu12_DZ, bit>,
    llama::Field<HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05, bit>,
    llama::Field<HLT_DoubleMu2_Jpsi_DoubleTkMu0_Phi, bit>,
    llama::Field<HLT_DoubleMu3_DCA_PFMET50_PFMHT60, bit>,
    llama::Field<HLT_TripleMu_5_3_3_Mass3p8_DCA, bit>,
    llama::Field<HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1, bit>,
    llama::Field<HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1, bit>,
    llama::Field<HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1, bit>,
    llama::Field<HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2, bit>,
    llama::Field<HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2, bit>,
    llama::Field<HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2, bit>,
    llama::Field<HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2, bit>,
    llama::Field<HLT_QuadPFJet98_83_71_15, bit>,
    llama::Field<HLT_QuadPFJet103_88_75_15, bit>,
    llama::Field<HLT_QuadPFJet105_88_76_15, bit>,
    llama::Field<HLT_QuadPFJet111_90_80_15, bit>,
    llama::Field<HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17, bit>,
    llama::Field<HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1, bit>,
    llama::Field<HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02, bit>,
    llama::Field<HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2, bit>,
    llama::Field<HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4, bit>,
    llama::Field<HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55, bit>,
    llama::Field<HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto, bit>,
    llama::Field<HLT_Mu12_IP6_part0, bit>,
    llama::Field<HLT_Mu12_IP6_part1, bit>,
    llama::Field<HLT_Mu12_IP6_part2, bit>,
    llama::Field<HLT_Mu12_IP6_part3, bit>,
    llama::Field<HLT_Mu12_IP6_part4, bit>,
    llama::Field<HLT_Mu9_IP5_part0, bit>,
    llama::Field<HLT_Mu9_IP5_part1, bit>,
    llama::Field<HLT_Mu9_IP5_part2, bit>,
    llama::Field<HLT_Mu9_IP5_part3, bit>,
    llama::Field<HLT_Mu9_IP5_part4, bit>,
    llama::Field<HLT_Mu7_IP4_part0, bit>,
    llama::Field<HLT_Mu7_IP4_part1, bit>,
    llama::Field<HLT_Mu7_IP4_part2, bit>,
    llama::Field<HLT_Mu7_IP4_part3, bit>,
    llama::Field<HLT_Mu7_IP4_part4, bit>,
    llama::Field<HLT_Mu9_IP4_part0, bit>,
    llama::Field<HLT_Mu9_IP4_part1, bit>,
    llama::Field<HLT_Mu9_IP4_part2, bit>,
    llama::Field<HLT_Mu9_IP4_part3, bit>,
    llama::Field<HLT_Mu9_IP4_part4, bit>,
    llama::Field<HLT_Mu8_IP5_part0, bit>,
    llama::Field<HLT_Mu8_IP5_part1, bit>,
    llama::Field<HLT_Mu8_IP5_part2, bit>,
    llama::Field<HLT_Mu8_IP5_part3, bit>,
    llama::Field<HLT_Mu8_IP5_part4, bit>,
    llama::Field<HLT_Mu8_IP6_part0, bit>,
    llama::Field<HLT_Mu8_IP6_part1, bit>,
    llama::Field<HLT_Mu8_IP6_part2, bit>,
    llama::Field<HLT_Mu8_IP6_part3, bit>,
    llama::Field<HLT_Mu8_IP6_part4, bit>,
    llama::Field<HLT_Mu9_IP6_part0, bit>,
    llama::Field<HLT_Mu9_IP6_part1, bit>,
    llama::Field<HLT_Mu9_IP6_part2, bit>,
    llama::Field<HLT_Mu9_IP6_part3, bit>,
    llama::Field<HLT_Mu9_IP6_part4, bit>,
    llama::Field<HLT_Mu8_IP3_part0, bit>,
    llama::Field<HLT_Mu8_IP3_part1, bit>,
    llama::Field<HLT_Mu8_IP3_part2, bit>,
    llama::Field<HLT_Mu8_IP3_part3, bit>,
    llama::Field<HLT_Mu8_IP3_part4, bit>,
    llama::Field<HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1, bit>,
    llama::Field<HLT_TrkMu6NoFiltersNoVtx, bit>,
    llama::Field<HLT_TrkMu16NoFiltersNoVtx, bit>,
    llama::Field<HLT_DoubleTrkMu_16_6_NoFiltersNoVtx, bit>,
    llama::Field<HLTriggerFinalPath, bit>,
    llama::Field<Flag_HBHENoiseFilter, bit>,
    llama::Field<Flag_HBHENoiseIsoFilter, bit>,
    llama::Field<Flag_CSCTightHaloFilter, bit>,
    llama::Field<Flag_CSCTightHaloTrkMuUnvetoFilter, bit>,
    llama::Field<Flag_CSCTightHalo2015Filter, bit>,
    llama::Field<Flag_globalTightHalo2016Filter, bit>,
    llama::Field<Flag_globalSuperTightHalo2016Filter, bit>,
    llama::Field<Flag_HcalStripHaloFilter, bit>,
    llama::Field<Flag_hcalLaserEventFilter, bit>,
    llama::Field<Flag_EcalDeadCellTriggerPrimitiveFilter, bit>,
    llama::Field<Flag_EcalDeadCellBoundaryEnergyFilter, bit>,
    llama::Field<Flag_ecalBadCalibFilter, bit>,
    llama::Field<Flag_goodVertices, bit>,
    llama::Field<Flag_eeBadScFilter, bit>,
    llama::Field<Flag_ecalLaserCorrFilter, bit>,
    llama::Field<Flag_trkPOGFilters, bit>,
    llama::Field<Flag_chargedHadronTrackResolutionFilter, bit>,
    llama::Field<Flag_muonBadTrackFilter, bit>,
    llama::Field<Flag_BadChargedCandidateFilter, bit>,
    llama::Field<Flag_BadPFMuonFilter, bit>,
    llama::Field<Flag_BadChargedCandidateSummer16Filter, bit>,
    llama::Field<Flag_BadPFMuonSummer16Filter, bit>,
    llama::Field<Flag_trkPOG_manystripclus53X, bit>,
    llama::Field<Flag_trkPOG_toomanystripclus53X, bit>,
    llama::Field<Flag_trkPOG_logErrorTooManyClusters, bit>,
    llama::Field<Flag_METFilters, bit>,
    llama::Field<L1Reco_step, bit>,
    llama::Field<L1_AlwaysTrue, bit>,
    llama::Field<L1_BPTX_AND_Ref1_VME, bit>,
    llama::Field<L1_BPTX_AND_Ref3_VME, bit>,
    llama::Field<L1_BPTX_AND_Ref4_VME, bit>,
    llama::Field<L1_BPTX_BeamGas_B1_VME, bit>,
    llama::Field<L1_BPTX_BeamGas_B2_VME, bit>,
    llama::Field<L1_BPTX_BeamGas_Ref1_VME, bit>,
    llama::Field<L1_BPTX_BeamGas_Ref2_VME, bit>,
    llama::Field<L1_BPTX_NotOR_VME, bit>,
    llama::Field<L1_BPTX_OR_Ref3_VME, bit>,
    llama::Field<L1_BPTX_OR_Ref4_VME, bit>,
    llama::Field<L1_BPTX_RefAND_VME, bit>,
    llama::Field<L1_BptxMinus, bit>,
    llama::Field<L1_BptxOR, bit>,
    llama::Field<L1_BptxPlus, bit>,
    llama::Field<L1_BptxXOR, bit>,
    llama::Field<L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142, bit>,
    llama::Field<L1_DoubleEG8er2p5_HTT260er, bit>,
    llama::Field<L1_DoubleEG8er2p5_HTT280er, bit>,
    llama::Field<L1_DoubleEG8er2p5_HTT300er, bit>,
    llama::Field<L1_DoubleEG8er2p5_HTT320er, bit>,
    llama::Field<L1_DoubleEG8er2p5_HTT340er, bit>,
    llama::Field<L1_DoubleEG_15_10_er2p5, bit>,
    llama::Field<L1_DoubleEG_20_10_er2p5, bit>,
    llama::Field<L1_DoubleEG_22_10_er2p5, bit>,
    llama::Field<L1_DoubleEG_25_12_er2p5, bit>,
    llama::Field<L1_DoubleEG_25_14_er2p5, bit>,
    llama::Field<L1_DoubleEG_27_14_er2p5, bit>,
    llama::Field<L1_DoubleEG_LooseIso20_10_er2p5, bit>,
    llama::Field<L1_DoubleEG_LooseIso22_10_er2p5, bit>,
    llama::Field<L1_DoubleEG_LooseIso22_12_er2p5, bit>,
    llama::Field<L1_DoubleEG_LooseIso25_12_er2p5, bit>,
    llama::Field<L1_DoubleIsoTau32er2p1, bit>,
    llama::Field<L1_DoubleIsoTau34er2p1, bit>,
    llama::Field<L1_DoubleIsoTau36er2p1, bit>,
    llama::Field<L1_DoubleJet100er2p3_dEta_Max1p6, bit>,
    llama::Field<L1_DoubleJet100er2p5, bit>,
    llama::Field<L1_DoubleJet112er2p3_dEta_Max1p6, bit>,
    llama::Field<L1_DoubleJet120er2p5, bit>,
    llama::Field<L1_DoubleJet150er2p5, bit>,
    llama::Field<L1_DoubleJet30er2p5_Mass_Min150_dEta_Max1p5, bit>,
    llama::Field<L1_DoubleJet30er2p5_Mass_Min200_dEta_Max1p5, bit>,
    llama::Field<L1_DoubleJet30er2p5_Mass_Min250_dEta_Max1p5, bit>,
    llama::Field<L1_DoubleJet30er2p5_Mass_Min300_dEta_Max1p5, bit>,
    llama::Field<L1_DoubleJet30er2p5_Mass_Min330_dEta_Max1p5, bit>,
    llama::Field<L1_DoubleJet30er2p5_Mass_Min360_dEta_Max1p5, bit>,
    llama::Field<L1_DoubleJet35_Mass_Min450_IsoTau45_RmOvlp, bit>,
    llama::Field<L1_DoubleJet40er2p5, bit>,
    llama::Field<L1_DoubleJet_100_30_DoubleJet30_Mass_Min620, bit>,
    llama::Field<L1_DoubleJet_110_35_DoubleJet35_Mass_Min620, bit>,
    llama::Field<L1_DoubleJet_115_40_DoubleJet40_Mass_Min620, bit>,
    llama::Field<L1_DoubleJet_115_40_DoubleJet40_Mass_Min620_Jet60TT28, bit>,
    llama::Field<L1_DoubleJet_120_45_DoubleJet45_Mass_Min620, bit>,
    llama::Field<L1_DoubleJet_120_45_DoubleJet45_Mass_Min620_Jet60TT28, bit>,
    llama::Field<L1_DoubleJet_80_30_Mass_Min420_DoubleMu0_SQ, bit>,
    llama::Field<L1_DoubleJet_80_30_Mass_Min420_IsoTau40_RmOvlp, bit>,
    llama::Field<L1_DoubleJet_80_30_Mass_Min420_Mu8, bit>,
    llama::Field<L1_DoubleJet_90_30_DoubleJet30_Mass_Min620, bit>,
    llama::Field<L1_DoubleLooseIsoEG22er2p1, bit>,
    llama::Field<L1_DoubleLooseIsoEG24er2p1, bit>,
    llama::Field<L1_DoubleMu0, bit>,
    llama::Field<L1_DoubleMu0_Mass_Min1, bit>,
    llama::Field<L1_DoubleMu0_OQ, bit>,
    llama::Field<L1_DoubleMu0_SQ, bit>,
    llama::Field<L1_DoubleMu0_SQ_OS, bit>,
    llama::Field<L1_DoubleMu0_dR_Max1p6_Jet90er2p5_dR_Max0p8, bit>,
    llama::Field<L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4, bit>,
    llama::Field<L1_DoubleMu0er1p5_SQ, bit>,
    llama::Field<L1_DoubleMu0er1p5_SQ_OS, bit>,
    llama::Field<L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4, bit>,
    llama::Field<L1_DoubleMu0er1p5_SQ_dR_Max1p4, bit>,
    llama::Field<L1_DoubleMu0er2p0_SQ_OS_dR_Max1p4, bit>,
    llama::Field<L1_DoubleMu0er2p0_SQ_dR_Max1p4, bit>,
    llama::Field<L1_DoubleMu10_SQ, bit>,
    llama::Field<L1_DoubleMu18er2p1, bit>,
    llama::Field<L1_DoubleMu3_OS_DoubleEG7p5Upsilon, bit>,
    llama::Field<L1_DoubleMu3_SQ_ETMHF50_HTT60er, bit>,
    llama::Field<L1_DoubleMu3_SQ_ETMHF50_Jet60er2p5, bit>,
    llama::Field<L1_DoubleMu3_SQ_ETMHF50_Jet60er2p5_OR_DoubleJet40er2p5, bit>,
    llama::Field<L1_DoubleMu3_SQ_ETMHF60_Jet60er2p5, bit>,
    llama::Field<L1_DoubleMu3_SQ_HTT220er, bit>,
    llama::Field<L1_DoubleMu3_SQ_HTT240er, bit>,
    llama::Field<L1_DoubleMu3_SQ_HTT260er, bit>,
    llama::Field<L1_DoubleMu3_dR_Max1p6_Jet90er2p5_dR_Max0p8, bit>,
    llama::Field<L1_DoubleMu4_SQ_EG9er2p5, bit>,
    llama::Field<L1_DoubleMu4_SQ_OS, bit>,
    llama::Field<L1_DoubleMu4_SQ_OS_dR_Max1p2, bit>,
    llama::Field<L1_DoubleMu4p5_SQ_OS, bit>,
    llama::Field<L1_DoubleMu4p5_SQ_OS_dR_Max1p2, bit>,
    llama::Field<L1_DoubleMu4p5er2p0_SQ_OS, bit>,
    llama::Field<L1_DoubleMu4p5er2p0_SQ_OS_Mass7to18, bit>,
    llama::Field<L1_DoubleMu5Upsilon_OS_DoubleEG3, bit>,
    llama::Field<L1_DoubleMu5_SQ_EG9er2p5, bit>,
    llama::Field<L1_DoubleMu9_SQ, bit>,
    llama::Field<L1_DoubleMu_12_5, bit>,
    llama::Field<L1_DoubleMu_15_5_SQ, bit>,
    llama::Field<L1_DoubleMu_15_7, bit>,
    llama::Field<L1_DoubleMu_15_7_Mass_Min1, bit>,
    llama::Field<L1_DoubleMu_15_7_SQ, bit>,
    llama::Field<L1_DoubleTau70er2p1, bit>,
    llama::Field<L1_ETM120, bit>,
    llama::Field<L1_ETM150, bit>,
    llama::Field<L1_ETMHF100, bit>,
    llama::Field<L1_ETMHF100_HTT60er, bit>,
    llama::Field<L1_ETMHF110, bit>,
    llama::Field<L1_ETMHF110_HTT60er, bit>,
    llama::Field<L1_ETMHF110_HTT60er_NotSecondBunchInTrain, bit>,
    llama::Field<L1_ETMHF120, bit>,
    llama::Field<L1_ETMHF120_HTT60er, bit>,
    llama::Field<L1_ETMHF120_NotSecondBunchInTrain, bit>,
    llama::Field<L1_ETMHF130, bit>,
    llama::Field<L1_ETMHF130_HTT60er, bit>,
    llama::Field<L1_ETMHF140, bit>,
    llama::Field<L1_ETMHF150, bit>,
    llama::Field<L1_ETMHF90_HTT60er, bit>,
    llama::Field<L1_ETT1200, bit>,
    llama::Field<L1_ETT1600, bit>,
    llama::Field<L1_ETT2000, bit>,
    llama::Field<L1_FirstBunchAfterTrain, bit>,
    llama::Field<L1_FirstBunchBeforeTrain, bit>,
    llama::Field<L1_FirstBunchInTrain, bit>,
    llama::Field<L1_FirstCollisionInOrbit, bit>,
    llama::Field<L1_FirstCollisionInTrain, bit>,
    llama::Field<L1_HCAL_LaserMon_Trig, bit>,
    llama::Field<L1_HCAL_LaserMon_Veto, bit>,
    llama::Field<L1_HTT120er, bit>,
    llama::Field<L1_HTT160er, bit>,
    llama::Field<L1_HTT200er, bit>,
    llama::Field<L1_HTT255er, bit>,
    llama::Field<L1_HTT280er, bit>,
    llama::Field<L1_HTT280er_QuadJet_70_55_40_35_er2p4, bit>,
    llama::Field<L1_HTT320er, bit>,
    llama::Field<L1_HTT320er_QuadJet_70_55_40_40_er2p4, bit>,
    llama::Field<L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3, bit>,
    llama::Field<L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3, bit>,
    llama::Field<L1_HTT360er, bit>,
    llama::Field<L1_HTT400er, bit>,
    llama::Field<L1_HTT450er, bit>,
    llama::Field<L1_IsoEG32er2p5_Mt40, bit>,
    llama::Field<L1_IsoEG32er2p5_Mt44, bit>,
    llama::Field<L1_IsoEG32er2p5_Mt48, bit>,
    llama::Field<L1_IsoTau40er2p1_ETMHF100, bit>,
    llama::Field<L1_IsoTau40er2p1_ETMHF110, bit>,
    llama::Field<L1_IsoTau40er2p1_ETMHF120, bit>,
    llama::Field<L1_IsoTau40er2p1_ETMHF90, bit>,
    llama::Field<L1_IsolatedBunch, bit>,
    llama::Field<L1_LastBunchInTrain, bit>,
    llama::Field<L1_LastCollisionInTrain, bit>,
    llama::Field<L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3, bit>,
    llama::Field<L1_LooseIsoEG22er2p1_Tau70er2p1_dR_Min0p3, bit>,
    llama::Field<L1_LooseIsoEG24er2p1_HTT100er, bit>,
    llama::Field<L1_LooseIsoEG24er2p1_IsoTau27er2p1_dR_Min0p3, bit>,
    llama::Field<L1_LooseIsoEG26er2p1_HTT100er, bit>,
    llama::Field<L1_LooseIsoEG26er2p1_Jet34er2p5_dR_Min0p3, bit>,
    llama::Field<L1_LooseIsoEG28er2p1_HTT100er, bit>,
    llama::Field<L1_LooseIsoEG28er2p1_Jet34er2p5_dR_Min0p3, bit>,
    llama::Field<L1_LooseIsoEG30er2p1_HTT100er, bit>,
    llama::Field<L1_LooseIsoEG30er2p1_Jet34er2p5_dR_Min0p3, bit>,
    llama::Field<L1_MinimumBiasHF0_AND_BptxAND, bit>,
    llama::Field<L1_Mu10er2p3_Jet32er2p3_dR_Max0p4_DoubleJet32er2p3_dEta_Max1p6, bit>,
    llama::Field<L1_Mu12er2p3_Jet40er2p1_dR_Max0p4_DoubleJet40er2p1_dEta_Max1p6, bit>,
    llama::Field<L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6, bit>,
    llama::Field<L1_Mu18er2p1_Tau24er2p1, bit>,
    llama::Field<L1_Mu18er2p1_Tau26er2p1, bit>,
    llama::Field<L1_Mu20_EG10er2p5, bit>,
    llama::Field<L1_Mu22er2p1_IsoTau32er2p1, bit>,
    llama::Field<L1_Mu22er2p1_IsoTau34er2p1, bit>,
    llama::Field<L1_Mu22er2p1_IsoTau36er2p1, bit>,
    llama::Field<L1_Mu22er2p1_IsoTau40er2p1, bit>,
    llama::Field<L1_Mu22er2p1_Tau70er2p1, bit>,
    llama::Field<L1_Mu3_Jet120er2p5_dR_Max0p4, bit>,
    llama::Field<L1_Mu3_Jet120er2p5_dR_Max0p8, bit>,
    llama::Field<L1_Mu3_Jet16er2p5_dR_Max0p4, bit>,
    llama::Field<L1_Mu3_Jet30er2p5, bit>,
    llama::Field<L1_Mu3_Jet35er2p5_dR_Max0p4, bit>,
    llama::Field<L1_Mu3_Jet60er2p5_dR_Max0p4, bit>,
    llama::Field<L1_Mu3_Jet80er2p5_dR_Max0p4, bit>,
    llama::Field<L1_Mu3er1p5_Jet100er2p5_ETMHF40, bit>,
    llama::Field<L1_Mu3er1p5_Jet100er2p5_ETMHF50, bit>,
    llama::Field<L1_Mu5_EG23er2p5, bit>,
    llama::Field<L1_Mu5_LooseIsoEG20er2p5, bit>,
    llama::Field<L1_Mu6_DoubleEG10er2p5, bit>,
    llama::Field<L1_Mu6_DoubleEG12er2p5, bit>,
    llama::Field<L1_Mu6_DoubleEG15er2p5, bit>,
    llama::Field<L1_Mu6_DoubleEG17er2p5, bit>,
    llama::Field<L1_Mu6_HTT240er, bit>,
    llama::Field<L1_Mu6_HTT250er, bit>,
    llama::Field<L1_Mu7_EG23er2p5, bit>,
    llama::Field<L1_Mu7_LooseIsoEG20er2p5, bit>,
    llama::Field<L1_Mu7_LooseIsoEG23er2p5, bit>,
    llama::Field<L1_NotBptxOR, bit>,
    llama::Field<L1_QuadJet36er2p5_IsoTau52er2p1, bit>,
    llama::Field<L1_QuadJet60er2p5, bit>,
    llama::Field<L1_QuadJet_95_75_65_20_DoubleJet_75_65_er2p5_Jet20_FWD3p0, bit>,
    llama::Field<L1_QuadMu0, bit>,
    llama::Field<L1_QuadMu0_OQ, bit>,
    llama::Field<L1_QuadMu0_SQ, bit>,
    llama::Field<L1_SecondBunchInTrain, bit>,
    llama::Field<L1_SecondLastBunchInTrain, bit>,
    llama::Field<L1_SingleEG10er2p5, bit>,
    llama::Field<L1_SingleEG15er2p5, bit>,
    llama::Field<L1_SingleEG26er2p5, bit>,
    llama::Field<L1_SingleEG34er2p5, bit>,
    llama::Field<L1_SingleEG36er2p5, bit>,
    llama::Field<L1_SingleEG38er2p5, bit>,
    llama::Field<L1_SingleEG40er2p5, bit>,
    llama::Field<L1_SingleEG42er2p5, bit>,
    llama::Field<L1_SingleEG45er2p5, bit>,
    llama::Field<L1_SingleEG50, bit>,
    llama::Field<L1_SingleEG60, bit>,
    llama::Field<L1_SingleEG8er2p5, bit>,
    llama::Field<L1_SingleIsoEG24er1p5, bit>,
    llama::Field<L1_SingleIsoEG24er2p1, bit>,
    llama::Field<L1_SingleIsoEG26er1p5, bit>,
    llama::Field<L1_SingleIsoEG26er2p1, bit>,
    llama::Field<L1_SingleIsoEG26er2p5, bit>,
    llama::Field<L1_SingleIsoEG28er1p5, bit>,
    llama::Field<L1_SingleIsoEG28er2p1, bit>,
    llama::Field<L1_SingleIsoEG28er2p5, bit>,
    llama::Field<L1_SingleIsoEG30er2p1, bit>,
    llama::Field<L1_SingleIsoEG30er2p5, bit>,
    llama::Field<L1_SingleIsoEG32er2p1, bit>,
    llama::Field<L1_SingleIsoEG32er2p5, bit>,
    llama::Field<L1_SingleIsoEG34er2p5, bit>,
    llama::Field<L1_SingleJet10erHE, bit>,
    llama::Field<L1_SingleJet120, bit>,
    llama::Field<L1_SingleJet120_FWD3p0, bit>,
    llama::Field<L1_SingleJet120er2p5, bit>,
    llama::Field<L1_SingleJet12erHE, bit>,
    llama::Field<L1_SingleJet140er2p5, bit>,
    llama::Field<L1_SingleJet140er2p5_ETMHF80, bit>,
    llama::Field<L1_SingleJet140er2p5_ETMHF90, bit>,
    llama::Field<L1_SingleJet160er2p5, bit>,
    llama::Field<L1_SingleJet180, bit>,
    llama::Field<L1_SingleJet180er2p5, bit>,
    llama::Field<L1_SingleJet200, bit>,
    llama::Field<L1_SingleJet20er2p5_NotBptxOR, bit>,
    llama::Field<L1_SingleJet20er2p5_NotBptxOR_3BX, bit>,
    llama::Field<L1_SingleJet35, bit>,
    llama::Field<L1_SingleJet35_FWD3p0, bit>,
    llama::Field<L1_SingleJet35er2p5, bit>,
    llama::Field<L1_SingleJet43er2p5_NotBptxOR_3BX, bit>,
    llama::Field<L1_SingleJet46er2p5_NotBptxOR_3BX, bit>,
    llama::Field<L1_SingleJet60, bit>,
    llama::Field<L1_SingleJet60_FWD3p0, bit>,
    llama::Field<L1_SingleJet60er2p5, bit>,
    llama::Field<L1_SingleJet8erHE, bit>,
    llama::Field<L1_SingleJet90, bit>,
    llama::Field<L1_SingleJet90_FWD3p0, bit>,
    llama::Field<L1_SingleJet90er2p5, bit>,
    llama::Field<L1_SingleLooseIsoEG28er1p5, bit>,
    llama::Field<L1_SingleLooseIsoEG30er1p5, bit>,
    llama::Field<L1_SingleMu0_BMTF, bit>,
    llama::Field<L1_SingleMu0_DQ, bit>,
    llama::Field<L1_SingleMu0_EMTF, bit>,
    llama::Field<L1_SingleMu0_OMTF, bit>,
    llama::Field<L1_SingleMu10er1p5, bit>,
    llama::Field<L1_SingleMu12_DQ_BMTF, bit>,
    llama::Field<L1_SingleMu12_DQ_EMTF, bit>,
    llama::Field<L1_SingleMu12_DQ_OMTF, bit>,
    llama::Field<L1_SingleMu12er1p5, bit>,
    llama::Field<L1_SingleMu14er1p5, bit>,
    llama::Field<L1_SingleMu15_DQ, bit>,
    llama::Field<L1_SingleMu16er1p5, bit>,
    llama::Field<L1_SingleMu18, bit>,
    llama::Field<L1_SingleMu18er1p5, bit>,
    llama::Field<L1_SingleMu20, bit>,
    llama::Field<L1_SingleMu22, bit>,
    llama::Field<L1_SingleMu22_BMTF, bit>,
    llama::Field<L1_SingleMu22_EMTF, bit>,
    llama::Field<L1_SingleMu22_OMTF, bit>,
    llama::Field<L1_SingleMu25, bit>,
    llama::Field<L1_SingleMu3, bit>,
    llama::Field<L1_SingleMu5, bit>,
    llama::Field<L1_SingleMu6er1p5, bit>,
    llama::Field<L1_SingleMu7, bit>,
    llama::Field<L1_SingleMu7_DQ, bit>,
    llama::Field<L1_SingleMu7er1p5, bit>,
    llama::Field<L1_SingleMu8er1p5, bit>,
    llama::Field<L1_SingleMu9er1p5, bit>,
    llama::Field<L1_SingleMuCosmics, bit>,
    llama::Field<L1_SingleMuCosmics_BMTF, bit>,
    llama::Field<L1_SingleMuCosmics_EMTF, bit>,
    llama::Field<L1_SingleMuCosmics_OMTF, bit>,
    llama::Field<L1_SingleMuOpen, bit>,
    llama::Field<L1_SingleMuOpen_NotBptxOR, bit>,
    llama::Field<L1_SingleMuOpen_er1p1_NotBptxOR_3BX, bit>,
    llama::Field<L1_SingleMuOpen_er1p4_NotBptxOR_3BX, bit>,
    llama::Field<L1_SingleTau120er2p1, bit>,
    llama::Field<L1_SingleTau130er2p1, bit>,
    llama::Field<L1_TOTEM_1, bit>,
    llama::Field<L1_TOTEM_2, bit>,
    llama::Field<L1_TOTEM_3, bit>,
    llama::Field<L1_TOTEM_4, bit>,
    llama::Field<L1_TripleEG16er2p5, bit>,
    llama::Field<L1_TripleEG_16_12_8_er2p5, bit>,
    llama::Field<L1_TripleEG_16_15_8_er2p5, bit>,
    llama::Field<L1_TripleEG_18_17_8_er2p5, bit>,
    llama::Field<L1_TripleEG_18_18_12_er2p5, bit>,
    llama::Field<L1_TripleJet_100_80_70_DoubleJet_80_70_er2p5, bit>,
    llama::Field<L1_TripleJet_105_85_75_DoubleJet_85_75_er2p5, bit>,
    llama::Field<L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5, bit>,
    llama::Field<L1_TripleMu0, bit>,
    llama::Field<L1_TripleMu0_OQ, bit>,
    llama::Field<L1_TripleMu0_SQ, bit>,
    llama::Field<L1_TripleMu3, bit>,
    llama::Field<L1_TripleMu3_SQ, bit>,
    llama::Field<L1_TripleMu_5SQ_3SQ_0OQ, bit>,
    llama::Field<L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu_5_3_SQ_OS_Mass_Max9, bit>,
    llama::Field<L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9, bit>,
    llama::Field<L1_TripleMu_5_3_3, bit>,
    llama::Field<L1_TripleMu_5_3_3_SQ, bit>,
    llama::Field<L1_TripleMu_5_3p5_2p5, bit>,
    llama::Field<L1_TripleMu_5_3p5_2p5_DoubleMu_5_2p5_OS_Mass_5to17, bit>,
    llama::Field<L1_TripleMu_5_3p5_2p5_OQ_DoubleMu_5_2p5_OQ_OS_Mass_5to17, bit>,
    llama::Field<L1_TripleMu_5_4_2p5_DoubleMu_5_2p5_OS_Mass_5to17, bit>,
    llama::Field<L1_TripleMu_5_5_3, bit>,
    llama::Field<L1_UnpairedBunchBptxMinus, bit>,
    llama::Field<L1_UnpairedBunchBptxPlus, bit>,
    llama::Field<L1_ZeroBias, bit>,
    llama::Field<L1_ZeroBias_copy, bit>
>;
// clang-format on
