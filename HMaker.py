import awkward as ak
import h5py
import numpy as np
import tqdm
import uproot
import argparse

DATA_STRUCTURE = np.dtype(
    [
        
        ("event","u4"),
        ("eventWeight","f4"),
        ("met","f4"),
        ("metphi","f4"),
        ("ptl1","f4"),
        ("etal1","f4"),
        ("phil1","f4"),
        ("nelectrons","u1"),
        ("njets","u1"),
        ("nlightjets","u1"),
        ("nbjets","f4"),
        ("pba1","f4"),
        ("pba2","f4"),
        ("pba3","f4"),
        ("pba4","f4"),
        ("pba5","f4"),
        ("pba6","f4"),
        ("pba7","f4"),
        ("pba8","f4"),
        ("pca1","f4"),
        ("pca2","f4"),
        ("pca3","f4"),
        ("pca4","f4"),
        ("pca5","f4"),
        ("pca6","f4"),
        ("pca7","f4"),
        ("pca8","f4"),
        ("pua1","f4"),
        ("pua2","f4"),
        ("pua3","f4"),
        ("pua4","f4"),
        ("pua5","f4"),
        ("pua6","f4"),
        ("pua7","f4"),
        ("pua8","f4"),
        ("ptaua1","f4"),
        ("ptaua2","f4"),
        ("ptaua3","f4"),
        ("ptaua4","f4"),
        ("ptaua5","f4"),
        ("ptaua6","f4"),
        ("ptaua7","f4"),
        ("ptaua8","f4"),
        ("pta1","f4"),
        ("pta2","f4"),
        ("pta3","f4"),
        ("pta4","f4"),
        ("pta5","f4"),
        ("pta6","f4"),
        ("pta7","f4"),
        ("pta8","f4"),
        ("etaa1","f4"),
        ("etaa2","f4"),
        ("etaa3","f4"),
        ("etaa4","f4"),
        ("etaa5","f4"),
        ("etaa6","f4"),
        ("etaa7","f4"),
        ("etaa8","f4"),
        ("phia1","f4"),
        ("phia2","f4"),
        ("phia3","f4"),
        ("phia4","f4"),
        ("phia5","f4"),
        ("phia6","f4"),
        ("phia7","f4"),
        ("phia8","f4"),
        ("ea1","f4"),
        ("ea2","f4"),
        ("ea3","f4"),
        ("ea4","f4"),
        ("ea5","f4"),
        ("ea6","f4"),
        ("ea7","f4"),
        ("ea8","f4"),

        ("dba1","f4"),
        ("dba2","f4"),
        ("dba3","f4"),
        ("dba4","f4"),
        ("dba5","f4"),
        ("dba6","f4"),
        ("dba7","f4"),
        ("dba8","f4"),
        ("dca1","f4"),
        ("dca2","f4"),
        ("dca3","f4"),
        ("dca4","f4"),
        ("dca5","f4"),
        ("dca6","f4"),
        ("dca7","f4"),
        ("dca8","f4"),
        ("tagbinscheme1a1","u1"),
        ("tagbinscheme1a2","u1"),
        ("tagbinscheme1a3","u1"),
        ("tagbinscheme1a4","u1"),
        ("tagbinscheme1a5","u1"),
        ("tagbinscheme1a6","u1"),
        ("tagbinscheme1a7","u1"),
        ("tagbinscheme1a8","u1"),
        ("tagbinscheme2a1","u1"),
        ("tagbinscheme2a2","u1"),
        ("tagbinscheme2a3","u1"),
        ("tagbinscheme2a4","u1"),
        ("tagbinscheme2a5","u1"),
        ("tagbinscheme2a6","u1"),
        ("tagbinscheme2a7","u1"),
        ("tagbinscheme2a8","u1"),
        ("tagbinscheme3a1","u1"),
        ("tagbinscheme3a2","u1"),
        ("tagbinscheme3a3","u1"),
        ("tagbinscheme3a4","u1"),
        ("tagbinscheme3a5","u1"),
        ("tagbinscheme3a6","u1"),
        ("tagbinscheme3a7","u1"),
        ("tagbinscheme3a8","u1"),
        ("tagbinscheme4a1","u1"),
        ("tagbinscheme4a2","u1"),
        ("tagbinscheme4a3","u1"),
        ("tagbinscheme4a4","u1"),
        ("tagbinscheme4a5","u1"),
        ("tagbinscheme4a6","u1"),
        ("tagbinscheme4a7","u1"),
        ("tagbinscheme4a8","u1"),
        ("tagbinscheme5a1","u1"),
        ("tagbinscheme5a2","u1"),
        ("tagbinscheme5a3","u1"),
        ("tagbinscheme5a4","u1"),
        ("tagbinscheme5a5","u1"),
        ("tagbinscheme5a6","u1"),
        ("tagbinscheme5a7","u1"),
        ("tagbinscheme5a8","u1"),
        ("tagbinscheme6a1","u1"),
        ("tagbinscheme6a2","u1"),
        ("tagbinscheme6a3","u1"),
        ("tagbinscheme6a4","u1"),
        ("tagbinscheme6a5","u1"),
        ("tagbinscheme6a6","u1"),
        ("tagbinscheme6a7","u1"),
        ("tagbinscheme6a8","u1"),
        ("tagbinscheme7a1","u1"),
        ("tagbinscheme7a2","u1"),
        ("tagbinscheme7a3","u1"),
        ("tagbinscheme7a4","u1"),
        ("tagbinscheme7a5","u1"),
        ("tagbinscheme7a6","u1"),
        ("tagbinscheme7a7","u1"),
        ("tagbinscheme7a8","u1"),        
    ]
)

def passes_selection(event, fraction):

    #choose a random sample of the events, based on fraction chosen and MET phi + lep phi
    if ((int)((event["metphi"]+event["phil1"])*10000) % 100 > 100*fraction): return False
    
    min_cjets = 1
    total_cjets = 0
    bthr = 0.844
    cthr = -0.725
    
    #Calculate jet discriminants
    #require 0, 1, or 2 c-tags of the loosest variety (70%)
    fc = 0.2
    fb = 0.3
    ftau_btag = 0.01
    ftau_ctag = 0.05
    fu_btag = 1-(fc+ftau_btag);
    fu_ctag = 1-(fb+ftau_ctag);
    
    dba1 = np.log(event["pba1"]/(fc*event["pca1"] + ftau_btag*event["ptaua1"] + fu_btag*event["pua1"]))
    dca1 = np.log(event["pca1"]/(fb*event["pba1"] + ftau_ctag*event["ptaua1"] + fu_ctag*event["pua1"]))
    if (dba1 < bthr and dca1 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    dba2 = np.log(event["pba2"]/(fc*event["pca2"] + ftau_btag*event["ptaua2"] + fu_btag*event["pua2"]))
    dca2 = np.log(event["pca2"]/(fb*event["pba2"] + ftau_ctag*event["ptaua2"] + fu_ctag*event["pua2"]))
    if (dba2 < bthr and dca2 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    dba3 = np.log(event["pba3"]/(fc*event["pca3"] + ftau_btag*event["ptaua3"] + fu_btag*event["pua3"]))
    dca3 = np.log(event["pca3"]/(fb*event["pba3"] + ftau_ctag*event["ptaua3"] + fu_ctag*event["pua3"]))
    if (dba3 < bthr and dca3 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    dba4 = np.log(event["pba4"]/(fc*event["pca4"] + ftau_btag*event["ptaua4"] + fu_btag*event["pua4"]))
    dca4 = np.log(event["pca4"]/(fb*event["pba4"] + ftau_ctag*event["ptaua4"] + fu_ctag*event["pua4"]))
    if (dba4 < bthr and dca4 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    dba5 = np.log(event["pba5"]/(fc*event["pca5"] + ftau_btag*event["ptaua5"] + fu_btag*event["pua5"]))
    dca5 = np.log(event["pca5"]/(fb*event["pba5"] + ftau_ctag*event["ptaua5"] + fu_ctag*event["pua5"]))
    if (dba5 < bthr and dca5 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    if not event["pca6"] > 0: return False
    dba6 = np.log(event["pba6"]/(fc*event["pca6"] + ftau_btag*event["ptaua6"] + fu_btag*event["pua6"]))
    dca6 = np.log(event["pca6"]/(fb*event["pba6"] + ftau_ctag*event["ptaua6"] + fu_ctag*event["pua6"]))
    if (dba6 < bthr and dca6 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    if not event["pca7"] > 0: return False
    dba7 = np.log(event["pba7"]/(fc*event["pca7"] + ftau_btag*event["ptaua7"] + fu_btag*event["pua7"]))
    dca7 = np.log(event["pca7"]/(fb*event["pba7"] + ftau_ctag*event["ptaua7"] + fu_ctag*event["pua7"]))
    if (dba7 < bthr and dca7 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True

    if not event["pca8"] > 0: return False
    dba8 = np.log(event["pba8"]/(fc*event["pca8"] + ftau_btag*event["ptaua8"] + fu_btag*event["pua8"]))
    dca8 = np.log(event["pca8"]/(fb*event["pba8"] + ftau_ctag*event["ptaua8"] + fu_ctag*event["pua8"]))
    if (dba8 < bthr and dca8 > cthr): total_cjets = total_cjets + 1
    if total_cjets >= min_cjets: return True
    
    return False

def gettagbin( db, dc, scheme):

    if scheme == 1:
        if db > 2.669: return 1
        if db > 1.892: return 2
        if db > 0.844: return 3
        if dc > 3.958: return 4
        if dc > 2.090: return 5
        if dc > 0.503: return 6
        return 7
    
    if scheme == 2:
        if db > 5.9: return 1
        if db > 4.3: return 2
        if db > 2.669: return 3
        if db > 1.892: return 4
        if db > 0.844: return 5
        if dc > 3.958: return 6
        if dc > 2.090: return 7
        if dc > 0.503: return 8
        return 9
    
    if scheme == 3:
        if db > 2.669: return 1
        if db > 2.3: return 2
        if db > 1.892: return 3
        if db > 1.375: return 4
        if db > 0.844: return 5
        if dc > 3.958: return 6
        if dc > 2.090: return 7
        if dc > 0.503: return 8
        return 9

    if scheme == 4:
        if db > 2.669: return 1
        if db > 1.892: return 2
        if db > 0.844: return 3
        if dc > 4.95: return 4
        if dc > 4.375: return 5
        if dc > 3.958: return 6
        if dc > 2.090: return 7
        if dc > 0.503: return 8
        return 9

    if scheme == 5:
        if db > 2.669: return 1
        if db > 1.892: return 2
        if db > 0.844: return 3
        if dc > 3.958: return 4
        if dc > 2.85: return 5
        if dc > 2.090: return 6
        if dc > 1.125: return 7
        if dc > 0.503: return 8
        return 9

    if scheme == 6:
        if db > 2.669: return 1
        if db > 1.892: return 2
        if db > 0.844: return 3
        if dc > 3.958: return 4
        if dc > 2.090: return 5
        if dc > 0.503: return 6
        if dc > -0.25: return 7
        if dc > -0.725: return 8
        return 9

    if scheme == 7:
        if db > 5.9: return 1
        if db > 4.3: return 2
        if db > 2.669: return 3
        if db > 2.3: return 4
        if db > 1.892: return 5
        if db > 1.375: return 6
        if db > 0.844: return 7
        if dc > 4.95: return 8
        if dc > 4.375: return 9
        if dc > 3.958: return 10
        if dc > 2.85: return 11
        if dc > 2.090: return 12
        if dc > 1.125: return 13
        if dc > 0.503: return 14
        if dc > -0.25: return 15
        if dc > -0.725: return 16
        return 17
    
    return 1
    
def get_flattened_event(event):

    flattened_event = np.zeros(1, dtype=DATA_STRUCTURE)
        
    #Basic variables
    flattened_event["event"] = event["event"]
    flattened_event["eventWeight"] = event["eventWeight"]
    flattened_event["met"] = event["met"]
    flattened_event["metphi"] = event["metphi"]
    flattened_event["ptl1"] = event["ptl1"]
    flattened_event["etal1"] = event["etal1"]
    flattened_event["phil1"] = event["phil1"]
    flattened_event["nelectrons"] = event["nelectrons"]
    flattened_event["njets"] = event["njets"]
    flattened_event["nlightjets"] = event["nlightjets"]
    flattened_event["nbjets"] = event["nbjets"]
    flattened_event["pba1"] = event["pba1"]
    flattened_event["pba2"] = event["pba2"]
    flattened_event["pba3"] = event["pba3"]
    flattened_event["pba4"] = event["pba4"]
    flattened_event["pba5"] = event["pba5"]
    flattened_event["pba6"] = event["pba6"]
    flattened_event["pba7"] = event["pba7"]
    flattened_event["pba8"] = event["pba8"]
    flattened_event["pca1"] = event["pca1"]
    flattened_event["pca2"] = event["pca2"]
    flattened_event["pca3"] = event["pca3"]
    flattened_event["pca4"] = event["pca4"]
    flattened_event["pca5"] = event["pca5"]
    flattened_event["pca6"] = event["pca6"]
    flattened_event["pca7"] = event["pca7"]
    flattened_event["pca8"] = event["pca8"]
    flattened_event["pua1"] = event["pua1"]
    flattened_event["pua2"] = event["pua2"]
    flattened_event["pua3"] = event["pua3"]
    flattened_event["pua4"] = event["pua4"]
    flattened_event["pua5"] = event["pua5"]
    flattened_event["pua6"] = event["pua6"]
    flattened_event["pua7"] = event["pua7"]
    flattened_event["pua8"] = event["pua8"]
    flattened_event["ptaua1"] = event["ptaua1"]
    flattened_event["ptaua2"] = event["ptaua2"]
    flattened_event["ptaua3"] = event["ptaua3"]
    flattened_event["ptaua4"] = event["ptaua4"]
    flattened_event["ptaua5"] = event["ptaua5"]
    flattened_event["ptaua6"] = event["ptaua6"]
    flattened_event["ptaua7"] = event["ptaua7"]
    flattened_event["ptaua8"] = event["ptaua8"]
    flattened_event["pta1"] = event["pta1"]
    flattened_event["pta2"] = event["pta2"]
    flattened_event["pta3"] = event["pta3"]
    flattened_event["pta4"] = event["pta4"]
    flattened_event["pta5"] = event["pta5"]
    flattened_event["pta6"] = event["pta6"]
    flattened_event["pta7"] = event["pta7"]
    flattened_event["pta8"] = event["pta8"]
    flattened_event["etaa1"] = event["etaa1"]
    flattened_event["etaa2"] = event["etaa2"]
    flattened_event["etaa3"] = event["etaa3"]
    flattened_event["etaa4"] = event["etaa4"]
    flattened_event["etaa5"] = event["etaa5"]
    flattened_event["etaa6"] = event["etaa6"]
    flattened_event["etaa7"] = event["etaa7"]
    flattened_event["etaa8"] = event["etaa8"]
    flattened_event["phia1"] = event["phia1"]
    flattened_event["phia2"] = event["phia2"]
    flattened_event["phia3"] = event["phia3"]
    flattened_event["phia4"] = event["phia4"]
    flattened_event["phia5"] = event["phia5"]
    flattened_event["phia6"] = event["phia6"]
    flattened_event["phia7"] = event["phia7"]
    flattened_event["phia8"] = event["phia8"]
    flattened_event["ea1"] = event["ea1"]
    flattened_event["ea2"] = event["ea2"]
    flattened_event["ea3"] = event["ea3"]
    flattened_event["ea4"] = event["ea4"]
    flattened_event["ea5"] = event["ea5"]
    flattened_event["ea6"] = event["ea6"]
    flattened_event["ea7"] = event["ea7"]
    flattened_event["ea8"] = event["ea8"]

    #Calculate jet discriminants and bins
    fc = 0.2
    fb = 0.3
    ftau_btag = 0.01
    ftau_ctag = 0.05
    fu_btag = 1-(fc+ftau_btag);
    fu_ctag = 1-(fb+ftau_ctag);

    #jet 1
    dba1 = np.log(event["pba1"]/(fc*event["pca1"] + ftau_btag*event["ptaua1"] + fu_btag*event["pua1"]))
    dca1 = np.log(event["pca1"]/(fb*event["pba1"] + ftau_ctag*event["ptaua1"] + fu_ctag*event["pua1"]))
    flattened_event["dba1"] = dba1
    flattened_event["dca1"] = dca1
    tagbinscheme1a1 = gettagbin( dba1, dca1, 1)
    tagbinscheme2a1 = gettagbin( dba1, dca1, 2)
    tagbinscheme3a1 = gettagbin( dba1, dca1, 3)
    tagbinscheme4a1 = gettagbin( dba1, dca1, 4)
    tagbinscheme5a1 = gettagbin( dba1, dca1, 5)
    tagbinscheme6a1 = gettagbin( dba1, dca1, 6)
    tagbinscheme7a1 = gettagbin( dba1, dca1, 7)
    flattened_event["tagbinscheme1a1"] = tagbinscheme1a1
    flattened_event["tagbinscheme2a1"] = tagbinscheme2a1
    flattened_event["tagbinscheme3a1"] = tagbinscheme3a1    
    flattened_event["tagbinscheme4a1"] = tagbinscheme4a1
    flattened_event["tagbinscheme5a1"] = tagbinscheme5a1
    flattened_event["tagbinscheme6a1"] = tagbinscheme6a1
    flattened_event["tagbinscheme7a1"] = tagbinscheme7a1
    
    #jet 2
    dba2 = np.log(event["pba2"]/(fc*event["pca2"] + ftau_btag*event["ptaua2"] + fu_btag*event["pua2"]))
    dca2 = np.log(event["pca2"]/(fb*event["pba2"] + ftau_ctag*event["ptaua2"] + fu_ctag*event["pua2"]))
    flattened_event["dba2"] = dba2
    flattened_event["dca2"] = dca2
    tagbinscheme1a2 = gettagbin( dba2, dca2, 1)
    tagbinscheme2a2 = gettagbin( dba2, dca2, 2)
    tagbinscheme3a2 = gettagbin( dba2, dca2, 3)
    tagbinscheme4a2 = gettagbin( dba2, dca2, 4)
    tagbinscheme5a2 = gettagbin( dba2, dca2, 5)
    tagbinscheme6a2 = gettagbin( dba2, dca2, 6)
    tagbinscheme7a2 = gettagbin( dba2, dca2, 7)
    flattened_event["tagbinscheme1a2"] = tagbinscheme1a2
    flattened_event["tagbinscheme2a2"] = tagbinscheme2a2
    flattened_event["tagbinscheme3a2"] = tagbinscheme3a2    
    flattened_event["tagbinscheme4a2"] = tagbinscheme4a2
    flattened_event["tagbinscheme5a2"] = tagbinscheme5a2
    flattened_event["tagbinscheme6a2"] = tagbinscheme6a2
    flattened_event["tagbinscheme7a2"] = tagbinscheme7a2

    #jet 3
    dba3 = np.log(event["pba3"]/(fc*event["pca3"] + ftau_btag*event["ptaua3"] + fu_btag*event["pua3"]))
    dca3 = np.log(event["pca3"]/(fb*event["pba3"] + ftau_ctag*event["ptaua3"] + fu_ctag*event["pua3"]))
    flattened_event["dba3"] = dba3
    flattened_event["dca3"] = dca3
    tagbinscheme1a3 = gettagbin( dba3, dca3, 1)
    tagbinscheme2a3 = gettagbin( dba3, dca3, 2)
    tagbinscheme3a3 = gettagbin( dba3, dca3, 3)
    tagbinscheme4a3 = gettagbin( dba3, dca3, 4)
    tagbinscheme5a3 = gettagbin( dba3, dca3, 5)
    tagbinscheme6a3 = gettagbin( dba3, dca3, 6)
    tagbinscheme7a3 = gettagbin( dba3, dca3, 7)
    flattened_event["tagbinscheme1a3"] = tagbinscheme1a3
    flattened_event["tagbinscheme2a3"] = tagbinscheme2a3
    flattened_event["tagbinscheme3a3"] = tagbinscheme3a3    
    flattened_event["tagbinscheme4a3"] = tagbinscheme4a3
    flattened_event["tagbinscheme5a3"] = tagbinscheme5a3
    flattened_event["tagbinscheme6a3"] = tagbinscheme6a3
    flattened_event["tagbinscheme7a3"] = tagbinscheme7a3

    #jet 4
    dba4 = np.log(event["pba4"]/(fc*event["pca4"] + ftau_btag*event["ptaua4"] + fu_btag*event["pua4"]))
    dca4 = np.log(event["pca4"]/(fb*event["pba4"] + ftau_ctag*event["ptaua4"] + fu_ctag*event["pua4"]))
    flattened_event["dba4"] = dba4
    flattened_event["dca4"] = dca4
    tagbinscheme1a4 = gettagbin( dba4, dca4, 1)
    tagbinscheme2a4 = gettagbin( dba4, dca4, 2)
    tagbinscheme3a4 = gettagbin( dba4, dca4, 3)
    tagbinscheme4a4 = gettagbin( dba4, dca4, 4)
    tagbinscheme5a4 = gettagbin( dba4, dca4, 5)
    tagbinscheme6a4 = gettagbin( dba4, dca4, 6)
    tagbinscheme7a4 = gettagbin( dba4, dca4, 7)
    flattened_event["tagbinscheme1a4"] = tagbinscheme1a4
    flattened_event["tagbinscheme2a4"] = tagbinscheme2a4
    flattened_event["tagbinscheme3a4"] = tagbinscheme3a4    
    flattened_event["tagbinscheme4a4"] = tagbinscheme4a4
    flattened_event["tagbinscheme5a4"] = tagbinscheme5a4
    flattened_event["tagbinscheme6a4"] = tagbinscheme6a4
    flattened_event["tagbinscheme7a4"] = tagbinscheme7a4

    #jet 5
    dba5 = np.log(event["pba5"]/(fc*event["pca5"] + ftau_btag*event["ptaua5"] + fu_btag*event["pua5"]))
    dca5 = np.log(event["pca5"]/(fb*event["pba5"] + ftau_ctag*event["ptaua5"] + fu_ctag*event["pua5"]))
    flattened_event["dba5"] = dba5
    flattened_event["dca5"] = dca5
    tagbinscheme1a5 = gettagbin( dba5, dca5, 1)
    tagbinscheme2a5 = gettagbin( dba5, dca5, 2)
    tagbinscheme3a5 = gettagbin( dba5, dca5, 3)
    tagbinscheme4a5 = gettagbin( dba5, dca5, 4)
    tagbinscheme5a5 = gettagbin( dba5, dca5, 5)
    tagbinscheme6a5 = gettagbin( dba5, dca5, 6)
    tagbinscheme7a5 = gettagbin( dba5, dca5, 7)
    flattened_event["tagbinscheme1a5"] = tagbinscheme1a5
    flattened_event["tagbinscheme2a5"] = tagbinscheme2a5
    flattened_event["tagbinscheme3a5"] = tagbinscheme3a5    
    flattened_event["tagbinscheme4a5"] = tagbinscheme4a5
    flattened_event["tagbinscheme5a5"] = tagbinscheme5a5
    flattened_event["tagbinscheme6a5"] = tagbinscheme6a5
    flattened_event["tagbinscheme7a5"] = tagbinscheme7a5

    #jet 6
    if event["pca6"] > 0:
        dba6 = np.log(event["pba6"]/(fc*event["pca6"] + ftau_btag*event["ptaua6"] + fu_btag*event["pua6"]))
        dca6 = np.log(event["pca6"]/(fb*event["pba6"] + ftau_ctag*event["ptaua6"] + fu_ctag*event["pua6"]))
    else:
        dba6 = -9.
        dca6 = -9.
    flattened_event["dba6"] = dba6
    flattened_event["dca6"] = dca6
    tagbinscheme1a6 = gettagbin( dba6, dca6, 1)
    tagbinscheme2a6 = gettagbin( dba6, dca6, 2)
    tagbinscheme3a6 = gettagbin( dba6, dca6, 3)
    tagbinscheme4a6 = gettagbin( dba6, dca6, 4)
    tagbinscheme5a6 = gettagbin( dba6, dca6, 5)
    tagbinscheme6a6 = gettagbin( dba6, dca6, 6)
    tagbinscheme7a6 = gettagbin( dba6, dca6, 7)
    flattened_event["tagbinscheme1a6"] = tagbinscheme1a6
    flattened_event["tagbinscheme2a6"] = tagbinscheme2a6
    flattened_event["tagbinscheme3a6"] = tagbinscheme3a6    
    flattened_event["tagbinscheme4a6"] = tagbinscheme4a6
    flattened_event["tagbinscheme5a6"] = tagbinscheme5a6
    flattened_event["tagbinscheme6a6"] = tagbinscheme6a6
    flattened_event["tagbinscheme7a6"] = tagbinscheme7a6

    #jet 7
    if event["pca7"] > 0:
        dba7 = np.log(event["pba7"]/(fc*event["pca7"] + ftau_btag*event["ptaua7"] + fu_btag*event["pua7"]))
        dca7 = np.log(event["pca7"]/(fb*event["pba7"] + ftau_ctag*event["ptaua7"] + fu_ctag*event["pua7"]))
    else:
        dba7 = -9.
        dca7 = -9.
    flattened_event["dba7"] = dba7
    flattened_event["dca7"] = dca7
    tagbinscheme1a7 = gettagbin( dba7, dca7, 1)
    tagbinscheme2a7 = gettagbin( dba7, dca7, 2)
    tagbinscheme3a7 = gettagbin( dba7, dca7, 3)
    tagbinscheme4a7 = gettagbin( dba7, dca7, 4)
    tagbinscheme5a7 = gettagbin( dba7, dca7, 5)
    tagbinscheme6a7 = gettagbin( dba7, dca7, 6)
    tagbinscheme7a7 = gettagbin( dba7, dca7, 7)
    flattened_event["tagbinscheme1a7"] = tagbinscheme1a7
    flattened_event["tagbinscheme2a7"] = tagbinscheme2a7
    flattened_event["tagbinscheme3a7"] = tagbinscheme3a7    
    flattened_event["tagbinscheme4a7"] = tagbinscheme4a7
    flattened_event["tagbinscheme5a7"] = tagbinscheme5a7
    flattened_event["tagbinscheme6a7"] = tagbinscheme6a7
    flattened_event["tagbinscheme7a7"] = tagbinscheme7a7

    #jet 8
    if event["pca8"] > 0:
        dba8 = np.log(event["pba8"]/(fc*event["pca8"] + ftau_btag*event["ptaua8"] + fu_btag*event["pua8"]))
        dca8 = np.log(event["pca8"]/(fb*event["pba8"] + ftau_ctag*event["ptaua8"] + fu_ctag*event["pua8"]))
    else:
        dba8 = -9.
        dca8 = -9.
    flattened_event["dba8"] = dba8
    flattened_event["dca8"] = dca8
    tagbinscheme1a8 = gettagbin( dba8, dca8, 1)
    tagbinscheme2a8 = gettagbin( dba8, dca8, 2)
    tagbinscheme3a8 = gettagbin( dba8, dca8, 3)
    tagbinscheme4a8 = gettagbin( dba8, dca8, 4)
    tagbinscheme5a8 = gettagbin( dba8, dca8, 5)
    tagbinscheme6a8 = gettagbin( dba8, dca8, 6)
    tagbinscheme7a8 = gettagbin( dba8, dca8, 7)
    flattened_event["tagbinscheme1a8"] = tagbinscheme1a8
    flattened_event["tagbinscheme2a8"] = tagbinscheme2a8
    flattened_event["tagbinscheme3a8"] = tagbinscheme3a8    
    flattened_event["tagbinscheme4a8"] = tagbinscheme4a8
    flattened_event["tagbinscheme5a8"] = tagbinscheme5a8
    flattened_event["tagbinscheme6a8"] = tagbinscheme6a8
    flattened_event["tagbinscheme7a8"] = tagbinscheme7a8
    
    return flattened_event

def create_data(file_list: list, fraction: float):

    tree = uproot.concatenate(
        file_list,
        [
            "event",
            "eventWeight",
            "met",
            "metphi",
            "ptl1",
            "etal1",
            "phil1",
            "nelectrons",
            "njets",
            "nlightjets",
            "nbjets",
            "pba1",
            "pba2",
            "pba3",
            "pba4",
            "pba5",
            "pba6",
            "pba7",
            "pba8",
            "pca1",
            "pca2",
            "pca3",
            "pca4",
            "pca5",
            "pca6",
            "pca7",
            "pca8",
            "pua1",
            "pua2",
            "pua3",
            "pua4",
            "pua5",
            "pua6",
            "pua7",
            "pua8",
            "ptaua1",
            "ptaua2",
            "ptaua3",
            "ptaua4",
            "ptaua5",
            "ptaua6",
            "ptaua7",
            "ptaua8",
            "pta1",
            "pta2",
            "pta3",
            "pta4",
            "pta5",
            "pta6",
            "pta7",
            "pta8",
            "etaa1",
            "etaa2",
            "etaa3",
            "etaa4",
            "etaa5",
            "etaa6",
            "etaa7",
            "etaa8",
            "phia1",
            "phia2",
            "phia3",
            "phia4",
            "phia5",
            "phia6",
            "phia7",
            "phia8",
            "ea1",
            "ea2",
            "ea3",
            "ea4",
            "ea5",
            "ea6",
            "ea7",
            "ea8",
        ],
    )

    print('running on fraction of {}'.format(fraction))
    
    n_evt_tot = len(tree)
    n_evt_sel = 0

    data = np.zeros(n_evt_tot, dtype=DATA_STRUCTURE)

    print('Reading {} events.'.format(n_evt_tot))

    for n_evt in tqdm.tqdm(range(n_evt_tot)):
        event = tree[n_evt]
        if not passes_selection(event, fraction): continue
        data[n_evt_sel] = get_flattened_event(event)
        n_evt_sel += 1

    print('Selected {} events out of {} total = {}%%'.format(n_evt_sel,n_evt_tot,100*n_evt_sel/n_evt_tot))

    return data[:n_evt_sel]
            

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest = 'inputs', default = 'ttH_cc',
                        help = 'input file')
    parser.add_argument('-f', '--fraction', dest = 'fraction', default = 1.0, type = float,
                        help = 'fraction  of data to run on')

    args = parser.parse_args()

    
    print("running on "+args.inputs)
    data_dir = "/tmp/jkeller/v1/"
    files = data_dir+args.inputs+".Nominal.root:NOSYS/selected"

    data = create_data(files, args.fraction)
    
    with h5py.File("output_"+args.inputs+".h5", "w") as file:
        file.create_dataset("events", data=data, dtype=DATA_STRUCTURE)
    
    print('Wrote file')
    

if __name__ == "__main__":
    main()
