# HFFNet 
This project provides the code and results for 'HFFNet: Hierarchical Feature Fusion Network for Lightweight RGB-T Semantic Segmentation'

# Requirements
-  python 3.7/3.8
-  pytorch 2.0.0+cu117

# Segmentation maps and performace
 We provide segmentation results on the MFNet and PST900 datasets  under the `./model/` directory:

- `predicts/` (default for MFNet dataset)
- `predicts_PST900/` (for PST900 dataset)

# Training
1. Install [apex](https://github.com/NVIDIA/apex).
2. Download the datasets:
   - [MFNet dataset](https://pan.baidu.com/s/1NHGazP7pwgEM47SP_ljJPg) (extraction code: `3b9o`)
   - [PST900 dataset](https://pan.baidu.com/s/13xgwFfUbu8zNvkwJq2Ggug) (extraction code: `mp2h`)
3. Run `train.py` (default uses MFNet dataset).

**Note**: the main model is located in `./toolbox/models/hffnet.py`


# Pre-trained models
- Pre-trained  models are provided under the `./model/` directory:
  - `model.pth` (MFNet dataset)
  - `model_PST900.pth` (PST900 dataset)



If you encounter any problems with the code, want to report bugs, etc.

Please contact me at xlwu0023@163.com.
