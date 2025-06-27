FAIR Universe challenge
===

Processed dataset available here: 

EOS path: `/eos/user/j/jsandesa/EFT_workshop_data/input_data_for_LPC_workshop.tar.gz`
Download: [CERNbox link](https://cernbox.cern.ch/s/zebMtgCM0JmbRxm)

The objective of FAIR Universe is to build an open, large-compute-scale AI ecosystem for sharing datasets, training & fine-tuning large models, and benchmarks in HEP (particle physics and cosmology). 

They host challenges focusing on measuring and minimizing the effects of systematic uncertainties in HEP.

Dataset
--

The tabular dataset, hosted on Zenodo (https://zenodo.org/records/15131565) is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. 

Since these events undergo the Delphes tool to produce simulated detector measurements, we have the advantage of potentially unlimited MC samples for development of NSBI. 

The first FAIR Universe challenge provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - ideal challenge to test the sensitivty of NSBI techniques.

Example published ATLAS distribution of the $H\to \tau\tau$ process among other backgrounds. The FAIR Universe dataset uses a $\tau_{had} \tau_{lep}$ final state, like in the distribution below:

![Screenshot 2025-06-06 at 11 22 19 AM](https://github.com/user-attachments/assets/3107e69c-7071-4dcd-bb3d-01777ba93746)
