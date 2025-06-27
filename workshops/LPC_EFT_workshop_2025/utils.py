import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt

# Calculate the preselection observable from multi-class classification NN outputs
def calculate_preselection_observable(pred_NN_incl, 
                                      labels_dict, 
                                      signal_processes, 
                                      background_processes, 
                                      pre_factor_dict = {'htautau': 1.0, 'ttbar': 1.0, 'ztautau': 1.0}):

    signal_sum = np.sum(
        [pre_factor_dict[signal] * pred_NN_incl[:, labels_dict[signal]] for signal in signal_processes], axis=0
    )

    background_sum = np.sum(
        [pre_factor_dict[background] * pred_NN_incl[:, labels_dict[background]] for background in background_processes], axis=0
    )
    
    # the preselection score
    presel_score = np.log(signal_sum/background_sum)

    return presel_score


# Split dataset into different channels based on preselection score
def preselection_using_score(dataset, channel_selections):

    mask_channel = {}
    dataset_channel = {}

    for channel, selection_dict in channel_selections.items():

        lower_cut = selection_dict.get('lower_presel')
        upper_cut = selection_dict.get('upper_presel')

        if (lower_cut != -999) and (upper_cut != -999):
    
            mask_channel = (
                (dataset.presel_score >= lower_cut) &
                (dataset.presel_score <= upper_cut)
            )
    
        elif (lower_cut != -999):
            mask_channel = (dataset.presel_score >= lower_cut)
    
        elif (upper_cut != -999):
            mask_channel = (dataset.presel_score <= upper_cut)
    
    
        dataset_channel[channel]   = dataset[mask_channel].copy()

    return dataset_channel


def plot_kinematic_features(
    features,
    nbins,
    dataset,
    xlabel_dict,
    labels_dict
):
    """
    For each feature in `features`, builds weighted histograms for each (label, variation)
    and then plots them on a grid of subplots.

    Parameters
    ----------
    features : list of str
        List of dataframe‐column names to histogram.
    nbins : int
        Number of bins to use (based on the 'nominal' variation of each feature).
    dataset : A DataFrame containing the data, which must have columns:
            - feature column in `features`
            - "detailed_labels" (to mask by label)
            - "weights"
    xlabel_dict : dict[str->str]
        Mapping each feature name → label for the x‐axis.
    labels_dict : list (or iterable) of str
        The list of unique labels to loop over (used to mask `df.detailed_labels == label`).

    Returns
    -------
    fig, axes
        The Figure and array of Axes where the histograms were drawn.
    """

    hist = {feat: {lbl: {} for lbl in labels_dict} for feat in features}
    bins_dict = {}

    for feature in features:
        
        kin_array = dataset[feature].to_numpy()
        bins = np.histogram_bin_edges(kin_array, bins=nbins)
        bins_dict[feature] = bins

        for label in labels_dict:
            mask = (dataset.detailed_labels == label)
            vals = dataset.loc[mask, feature].to_numpy()
            wts = dataset.loc[mask, 'weights'].to_numpy()
            hist_vals, _ = np.histogram(vals, weights=wts, bins=bins)
            hist[feature][label] = hist_vals

    n_plots = len(features)
    ncols = 2
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_label_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels_dict)}


    # Plot each feature in its own Axes
    for ax, feature in zip(axes, features):
        for label in labels_dict:
            hep.histplot(
                hist[feature][label],
                bins=bins_dict[feature],
                label=label,
                ax=ax,
                linewidth=1.5,
                color=color_label_map[label]
            )

        ax.set_yscale('log')
        ax.set_xlabel(xlabel_dict.get(feature, feature), size=14)
        ax.set_ylabel('Events', size=14)
        ax.legend()

    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig, axes





