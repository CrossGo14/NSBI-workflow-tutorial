import numpy as np

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