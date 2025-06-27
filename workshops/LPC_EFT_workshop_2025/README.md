FAIR Universe Dataset
--

In this tutorial, we will be making use of the publicly available dataset from the FAIR Universe challenge. 

<img width="592" alt="Screenshot 2025-06-27 at 4 53 03 PM" src="https://github.com/user-attachments/assets/830bcee3-5b1a-4411-be24-fd008696a112" />


The dataset we use provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - ideal challenge to test the sensitivty of NSBI techniques.

The tabular dataset, hosted on Zenodo (https://zenodo.org/records/15131565) is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. 

Example published ATLAS distribution of the $H\to \tau\tau$ process among other backgrounds. The FAIR Universe dataset uses a $\tau_{had} \tau_{lep}$ final state, like in the distribution below:

![Screenshot 2025-06-06 at 11 22 19 AM](https://github.com/user-attachments/assets/3107e69c-7071-4dcd-bb3d-01777ba93746)


Get Models and Processed Data
==

Processed dataset and NN models needed for this tutorial can be copied over from lxplus: 

```
scp <your_user_name>@lxplus.cern.ch:/eos/user/j/jsandesa/EFT_workshop_data/input_data_for_LPC_workshop.tar.gz ./LPC_EFT_workshop_2025/
```

If you do not have an lxplus account, you can download the dataset using the [CERNbox link](https://cernbox.cern.ch/s/zebMtgCM0JmbRxm).

Move the `input_data_for_LPC_workshop.tar.gz` to the `workshops/LPC_EFT_workshop_2025/` directory if it is not already there, and do:

```
tar -xvf input_data_for_LPC_workshop.tar.gz
```
