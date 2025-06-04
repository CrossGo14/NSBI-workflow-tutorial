import numpy as np
import pandas as pd

def calculate_preselection_observable(pred_NN_incl, labels_dict, signal_processes, background_processes, pre_factor_dict = {'htautau': 1.0, 'ttbar': 1.0, 'ztautau': 1.0}):

    signal_sum = np.sum(
        [pre_factor_dict[signal] * pred_NN_incl[:, labels_dict[signal]] for signal in signal_processes], axis=0
    )

    background_sum = np.sum(
        [pre_factor_dict[background] * pred_NN_incl[:, labels_dict[background]] for background in background_processes], axis=0
    )
    # the preselection score as defined above - log(P_S/P_B)
    presel_score = np.log(signal_sum/background_sum)

    return presel_score


def preselection_using_score(dataset, preselection_cuts):

    lower_cut = preselection_cuts.get('lower')
    upper_cut = preselection_cuts.get('upper')
    
    masks = {
        'SR': pd.Series(True, index=dataset.index),
        'CR': pd.Series(False, index=dataset.index)
    }
    
    # CASE A: both lower and upper exist (and are “active”, i.e. not -999)
    if (lower_cut != -999) and (upper_cut != -999):
        masks['SR'] = (
            (dataset.presel_score >= lower_cut) &
            (dataset.presel_score <= upper_cut)
        )
        masks['CR'] = ~masks['SR']  
    
    # CASE B: only a lower‐side cut is active
    elif (lower_cut != -999):
        masks['SR'] = (dataset.presel_score >= lower_cut)
        masks['CR'] = (dataset.presel_score <  lower_cut)
    
    # CASE C: only an upper‐side cut is active
    elif (upper_cut != -999):
        masks['SR'] = (dataset.presel_score <= upper_cut)
        masks['CR'] = (dataset.presel_score >  upper_cut)
    
    
    dataset_SR   = dataset[masks['SR']].copy()
    dataset_CR = dataset[masks['CR']].copy()

    return dataset_SR, dataset_CR