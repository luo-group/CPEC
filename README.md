# CPEC 
Leveraging conformal prediction to annotate enzyme universe with limited false positives

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
In this repository, we provide an example of CPEC predicting EC numbers in notebook `demo.ipynb`. We provide the raw output of our base model in folder `example_data/` as an example. You can test the FDR-controlled (false discovery rate) EC number prediction entirely in this repository. Even through CPEC can be applied to general machine learning methods for prediction, we used [PenLight](https://github.com/luo-group/PenLight), a contrastive learning-based model in CPEC as an illustration. 

If you want to test other base models, you can normalize the predicted probabilities into $\left[0,1\right]$ and save the predicted probabilities and ground truths into tensors of the shape $\left[n_{samples}, n_{labels}\right]$. Then, using the function `calibrate_fdr()`, you can calculate the valid model parameter on calibration data and make FDR-controlled predictions on your own test data.  