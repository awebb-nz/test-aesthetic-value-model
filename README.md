# Testing a computational model of aesthetic value
Data and analysis files for an experiment testing the viability of [a computational model of aesthetic value developed by Brielmann and Dayan (2022)](https://psycnet.apa.org/fulltext/2022-78031-001.html)(go to psyarxiv for the [free preprint](https://psyarxiv.com/eaqkc/) if you do not have accesss).
Analyses and plotting functions used here depend on the [main package containing the core functions of the computational model of aesthetic value](https://github.com/aenneb/intro-aesthetic-value-model).

For convenience, this repo contains copies of the necessary core package files. So, you do not need to download any additional scripts from another repo and all paths are set so that you can run everything with this repo only. 

If you want to replicate all analyses, you can run all scripts located in the "analysis" folder in alphabetic order. However, I would recommend not to re-fit the model for all participants unless you have a substantial amount of time and/or processing power.

# Running the code

If you want run the code, try the following steps:
```
git clone https://github.com/aenneb/test-aesthetic-value-model.git 
cd test-aesthetic-value-model 
python -m venv .env 
.env/Scripts/activate # windows
# or 
source .env/bin/activate # mac/linux
pip install -r requirements.txt 
cd analysis 
python a_get_complete_data.py #etc
```

You should then be able to run any of the analysis scripts. You will also note the `requirements.txt` file which lists all packages that are required to run all scripts. NOTE that pingouin may not be compatible with the latest numpy version. To run the analyses that require pingouin, downgrade your environment's numpy version to < 1.24.

## Which code to run and what it does

???


# Folder content

The main directory contains the pre-processed data from all participants as well as the results of the simulated, random-order refits.

It also contains the two components of the core aesthetic value model you need to reproduce the analyses.

In addition, it contains "map_imgName_imgIdx.csv" which provides a safe way to ensure that DNN-derived image features are mapped unto the correct images in the data files. 

As a bonus, it contains the "plot_images_inVGGspace.py" script which visualizes the location of the stimulus images in reduced DNN-feature space.


## analysis

Contains all scripts needed to reproduce the analyses reported in the paper as well as the code for re-creating the figures. 

NOTE that the model fitting may take a substantial amount of time, so if you do not want to tweak this, you might want to use the available model fitting results from the "results" folder.

## data_experiment

Contains all raw data that was collected. 

NOTE that these files also contain participants that were excluded from analyses.

## results

Contains the .csv and .npy files for all preliminary and final results of the analyses that are the basis for statistical analyses and plots.

## experiment_code

Contains the complete code for running the experiment as described in the paper *except* for the core jspsych files. Download jspsych 6.3.1 and save the folder inside the experiment_code folder to run the experiment.

## vgg_features

Contains the feature values and their PCA-reduced version for all stimuli based on vanilla, pretrained VGG-16.

## ResNet50_features

Contains the feature values and their PCA-reduced version for all stimuli based on vanilla, pretrained ResNet50.

## figures

Contains the figures from the paper.


