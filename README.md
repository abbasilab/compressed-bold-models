# Compression-enabled interpretability of voxel-wise encoding models

Authors: Fatemeh Kamali, Amir Abolfazl Suratgar, Mohammadbagher Menhaj, Reza Abbasi-Asl

Manuscript link: https://www.biorxiv.org/content/10.1101/2022.07.09.494042v1.abstract


## 1. Extract Frames

Extract frames from videos (using, for example, ffmpeg) and save them in a folder. The frame size should be 227x227.

## 2. Feature Extraction

2.1 Run feature_extraction.py to extract UC features of each frame.
   
2.2 DC feature_extraction: Follow the steps in https://github.com/mightydeveloper/Deep-Compression-PyTorch to build DC caffemodel
then use it in feature extraction.py to extract DC features

2.3 Run RC_feature_extraction.py to compress using the receptive field approach.

2.4 Run SC_feature_extraction.py to structurally compress features.

## 3. Preprocess the features:
   
3.1 Run lw_preprocessing.m to compute principal components, convolve features with HRF and downsample them for each layer.
   
3.2 Run run all_preprocessing.m. to pre-process features for all-layer analysis.

## 4. Regression

Run regression.mat with the preprocessed features and the fMRI responses to obtain regression weights and predictive accuracy.

## 5. Interpretation

5.1 Run layer_contribution.py Layer to obtain contributaions of each layer to predictive performance.

5.2 Run identification.mat to obtain video clip identification performance.

5.3 To obtain top preferred images, first run feature_extraction.py to obtain features for each image in the ImagNet validation set. Then follow pre-processing steps above to obtain PCA dimensionality reduced data. Finally, use image_selection_alllayer.m to obtain top filters using the weights from the regression and the feaures.

5.4 Run plots.feat_cor.m to compute correlations between top preferred images 

5.5 Run plots.Filter_visualization.py to visualize filters

5.5 Run rcsrc.center_radius.py to obtain receptive filed size and locations

# Version

Last update: Mar 2023, v1.0

# Citation

Fatemeh Kamali, Amir Abolfazl Suratgar, Mohammadbagher Menhaj, and Reza Abbasi-Asl. "Compression-enabled interpretability of voxel-wise encoding models." bioRxiv (2022): 2022-07.

