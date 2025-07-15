Note: this is an outdated example, no longer supported by the latest version of `nsbi-common-utils`. Please look at the FAIR Universe example for a working tutorial.

# Application to CMS Open data - ttbar production

-----

This example needs the execution of notebooks in the following order, with intermediate data cached locally. 

1. [`Data_Preprocessing.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Data_Preprocessing.ipynb) - loads CMS open data using the `nanoaod_inputs.json` config file, and processes it into dataframe and numpy objects for subsequent likelihood ratio training step.
2. [`Neural_Likelihood_Ratio_Estimation.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Neural_Likelihood_Ratio_Estimation.ipynb) - loads cached dataframe object and trains density ratios using the nominal samples.
3. [`Systematic_Uncertainty_Estimation.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Systematic_Uncertainty_Estimation.ipynb) - loads cached dataframe object and trains density ratios for the systematic variation samples.
4. [`Parameter_Fitting_with_Systematics.ipynb`](https://github.com/iris-hep/NSBI-workflow-tutorial/blob/main/cms_ttbar_open_data/Parameter_Fitting_with_Systematics.ipynb) - uses the trained density ratios for unbinned statistical inference.
