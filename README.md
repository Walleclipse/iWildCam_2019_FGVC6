# iWildCam_2019_FGVC6
Top 3% (7/336)  solution for [iWildCam 2019](https://www.kaggle.com/c/iwildcam-2019-fgvc6/overview) competition (Categorize animals in the wild), which is as part of the  [FGVC6](https://sites.google.com/view/fgvc6/home) workshop at [CVPR 2019](http://cvpr2019.thecvf.com/)

Thanks to my team members!

### Requirements
* Python 3.6
* pytorch 1.1.0

### About the Code

#### 1. Prepare Data
Download the competition data according to [here](data/README.md)

After downloading, record the image-file in CSV format.
```
python prep_data.py
```

#### 2. Detect and Crop the Image 
```
python detect_crop_image.py
```
In my method, I first run object detection and crop the bounding box for classification. 
#### 3. Train the Model
```
python train_model.py
```
#### 4. Prediction

```
python infer.py
```

### About the Method

I got the best single model predicion result (f1=0.224 in private LB) with following configurations.

model: efficientnet_b0 (imagenet pretrained)

image augmentation: traditional image augmentation + CLAHE + gray scale + cutout + mixup + label smoothing

Please view the detailed report [iwildcam_2019_report.pdf](iwildcam_2019_report.pdf) or [Efficient Method for Categorize Animals in the Wild](https://arxiv.org/ftp/arxiv/papers/1907/1907.13037.pdf)
