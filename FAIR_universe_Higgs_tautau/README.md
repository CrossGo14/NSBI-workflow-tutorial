FAIR Universe Dataset
--

The tabular dataset used in this demonstration is hosted on Zenodo (https://zenodo.org/records/15131565), and is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The dataset provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - good challenge to test the sensitivty of NSBI techniques.

Workflow
--

1. [`DataLoader.ipynb`](./DataLoader.ipynb)

This notebook is to download the FAIR Universe data and store it in the form of `.root` ntuples. This step only needs to be done once and is independent of the overall NSBI workflow. 

2. [`Preselection_withNN.ipynb`](./Preselection_withNN.ipynb) 

3. [`DataPreprocessing.ipynb`](./DataPreprocessing.ipynb)

4. [`Neural_Likelihood_Ratio_Estimation.ipynb`](./Neural_Likelihood_Ratio_Estimation.ipynb)

5. [`Systematic_Uncertainty_Estimation.ipynb`](./Systematic_Uncertainty_Estimation.ipynb)

6. [`Parameter_Fitting_with_Systematics.ipynb`](./Parameter_Fitting_with_Systematics.ipynb)
   


Measurements at the LHC
--

Example published ATLAS distribution of the $H\to \tau\tau$ process among other backgrounds. The FAIR Universe dataset uses a $\tau_{had} \tau_{lep}$ final state, like in the distribution below:

![Screenshot 2025-06-06 at 11 22 19 AM](https://github.com/user-attachments/assets/3107e69c-7071-4dcd-bb3d-01777ba93746)
