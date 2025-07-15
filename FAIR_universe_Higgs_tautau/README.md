Workflow
--

This example needs the execution of notebooks in the following order, with intermediate data cached locally. 

1. [`Data_Preprocessing.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Data_Preprocessing.ipynb) - loads FAIR Universe dataset and stores a dataframe.
2. [`Neural_Likelihood_Ratio_Estimation.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Neural_Likelihood_Ratio_Estimation.ipynb) - loads cached dataframe object and trains density ratios using the nominal samples.
3. [`Systematic_Uncertainty_Estimation.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Systematic_Uncertainty_Estimation.ipynb) - loads cached dataframe object and trains density ratios for the systematic variation samples.
4. [`Parameter_Fitting_with_Systematics.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Parameter_Fitting_with_Systematics.ipynb) - uses the trained density ratios for unbinned statistical inference.

FAIR Universe Dataset
--

The tabular dataset used in this demonstration is hosted on Zenodo (https://zenodo.org/records/15131565), and is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The dataset provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - good challenge to test the sensitivty of NSBI techniques.

Example published ATLAS distribution of the $H\to \tau\tau$ process among other backgrounds. The FAIR Universe dataset uses a $\tau_{had} \tau_{lep}$ final state, like in the distribution below:

![Screenshot 2025-06-06 at 11 22 19 AM](https://github.com/user-attachments/assets/3107e69c-7071-4dcd-bb3d-01777ba93746)
