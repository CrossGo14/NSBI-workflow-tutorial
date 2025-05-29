# HEP-Challenge Program

This folder contains the ingestion program for the HEP-Challenge.

## Overview
The HEP-Challenge Program is responsible for processing and ingesting data for the HEP-Challenge. It provides a set of functionalities to handle data ingestion, transformation, running the participants' model and finally scoring the submissions. 

### Running Ingestion Program locally.
To run the ingestion in a CPU-only system use 
```bash
python3 run_ingestion.py \ 
--systematics-tes \ 
--systematics-soft-met \ 
--systematics-jes \ 
--systematics-ttbar-scale \ 
--systematics-diboson-scale \ 
--systematics-bkg-scale \
--num-pseudo-experiments 100 \ 
--num-of-sets 1 
```

If you have GPU you can use the `--parallel` flag to parallelize the pseudo experiments.

### Running Scoring Program locally.
To run the scoring program, use the following command:

```
python run_scoring.py 
```

Check out the flags of `run_scoring.py` by using `python run_scoring.py -h `


The detailed coverage plots will be available in the [detailed_results.html](/scoring_output/detailed_results.html)

For more information on the workings of the the HEP Challenge program check out our [whitepaper](https://fair-universe.lbl.gov/whitepaper.pdf)







