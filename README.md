# Encoding Model
Example code and analysis for the paper: Compression-enabled interpretability of voxelwise encoding models
o simulate results in the paper, follow these steps:

## 1. Extract Frames

Extract frames from videos (using, for example, ffmpeg) and save them in a folder. The frame size should be 227x227.

## 2. Feature Extraction

2.1 Run feature_extraction.py to extract UC features of each frame.
   
2.2 DC feature_extraction: Follow the steps in https://github.com/mightydeveloper/Deep-Compression-PyTorch to build DC caffemodel
then use it in feature extraction.py to extract DC features

2.3 Run RC_feature_extraction.py to compress using the receptive field approach.

2.4 Run SC_feature_extraction.py to structurally compress features.

## 3. Preprocess the features:
   
3.1 To compute principal components, convolve features with HRF and downsample them for each layer, run lw_preprocessing.m.
   
3.2 To preprocess features for all-layer analysis, run all_preprocessing.m.

## 4. Regression

Use the preprocessed features and the fMRI responses as the input of regression.mat to calculate weights and accuracy

## 5. Interpretation

5.1 Layer contributaions are computed using layer_contribution.py in plots folder

5.2 Video clip identification performance can be computed using identification.mat

5.3 To obtain top images that actiavte each voxel, first the features of each image in the validation set of the ImagNet must be calculated using feature_extraction.py. 
Then PCA-reduced demension should be calculated similar to previous steps. the weights obtained from regression and the feaures should be used as the input of image_selection_alllayer.m to obtain top filters.

5.4. to compute correlation between top images run plots.feat_cor.m

5.5 plots.Filter_visualization.py can be used to visualize the filters

5.5 Rreceptive filed size and locations can be obtained using rcsrc.center_radius.py


