# CPEC 
This is the official repository for the paper ***Leveraging conformal prediction to annotate enzyme function space with limited false positives***.

## Table of contents
- [CPEC](#cpec)
  - [Table of contents](#table-of-contents)
  - [Install dependencies](#install-dependencies)
  - [Run CPEC](#run-cpec)
    - [(Optional) Base model implementation](#optional-base-model-implementation)
      - [PenLight2](#penlight2)
      - [CLEAN](#clean)
      - [Other base models](#other-base-models)
    - [False Discovery Rate (FDR)-controlled EC number prediction](#false-discovery-rate-fdr-controlled-ec-number-prediction)


## Install dependencies
```
conda create -n cpec python=3.9
conda activate cpec
```
We use pytorch 1.12.1, which can be installed by following the instructions on their offical website.
```
pip install -r requirements.txt
```

## Run CPEC



### (Optional) Base model implementation

#### PenLight2
We provide the implementation of PenLight2 in [base_models/PenLight2](https://github.com/luo-group/CPEC/tree/base_models/base_models/PenLight2).

#### CLEAN
We provide the implementation of CLEAN in [base_models/CLEAN](https://github.com/luo-group/CPEC/tree/base_models/base_models/CLEAN).

#### Other base models

### False Discovery Rate (FDR)-controlled EC number prediction

In this repository, we provide an example of CPEC predicting EC numbers in notebook `demo.ipynb`. We provide the raw output of our base model in folder `example_data/` as an example. You can test the FDR-controlled (false discovery rate) EC number prediction entirely in this repository. Even though CPEC can be applied to general machine learning methods for prediction, we used [PenLight2](https://github.com/luo-group/PenLight), a contrastive learning-based model in CPEC as an illustration. 

If you want to test other base models, you can normalize the predicted probabilities into $\left[0,1\right]$ and save the predicted probabilities and ground truths into tensors of the shape $\left[n_{samples}, n_{labels}\right]$. Then, using the function `calibrate_fdr()`, you can calculate the valid model parameter on calibration data and make FDR-controlled predictions on your own test data.  