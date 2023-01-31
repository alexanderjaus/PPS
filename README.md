# PPS: Wild Panoramic Panoptic Segmentation dataset

## Google Drive
[Wild PPS dataset on GoogleDrive](https://drive.google.com/file/d/1juZRdSnO6Q7Xvt_UVjv_b-EuBtJhNybv/view?usp=sharing)
## Description
We provide a diverse dataset consisting of 80 panoramic images from 40 different cities taken from [WildPASS](https://github.com/elnino9ykl/WildPASS) along with panoptic annotations for the most essential street scence classes (Stuff: Road, Sidewalk & Thing: Person, Car) in [cityscapes](https://www.cityscapes-dataset.com) annotation format. 

---
<img src="Readme/Readme_Example.png" alt="logo">

## Trained Models
We provide our proposed robust seamless segmentation model. The weights can be downloaded from Google drive [Robust Seamless Segmentation Model](https://drive.google.com/file/d/1LUZPINWer0z2dr8iHW7xso6uCFnGoKo2/view?usp=sharing).

## Getting started with the robust seamless segmentation model
### Setup
- Follow the installation steps for the seamless segmentation model as described in the [official repository](https://github.com/mapillary/seamseg)
- Download the provided [weights](https://drive.google.com/file/d/1LUZPINWer0z2dr8iHW7xso6uCFnGoKo2/view?usp=sharing)
- Download the [WildPPS dataset](ttps://drive.google.com/file/d/1juZRdSnO6Q7Xvt_UVjv_b-EuBtJhNybv/view?usp=sharing). If you are using a custom dataset, make sure your dataset is in the official [cityscapes](https://www.cityscapes-dataset.com) data format. If it is not the case, use [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) to bring it to the required format.
### Running inference 
- Panoptic predictions can be computed as described in the [official seamseg repository](https://github.com/mapillary/seamseg). Make sure to use the ```--raw``` flag to compute quantitative results.
- Use the provided ```robust_seamseg.ini``` config file and update the path to  the weights in line 13
### Computing the Panoptic Quality
- Convert the provided WildPPS dataset into the required format for the seamless segmentation model with the help of this [script](https://github.com/mapillary/seamseg/blob/main/scripts/data_preparation/prepare_cityscapes.py) or download the precomputed [WildPPS_seamsegformat](https://drive.google.com/file/d/1zyG6Brlyk7aEGwmqvgSdxvj3wbrC-2N2/view?usp=sharing) dataset. This is only necessary if you use the seamless segmentation models. Other popular libraries such as [DETECTRON2](https://github.com/facebookresearch/detectron2) can directly work with the normal format.
- Use the provided ```measure_panoptic.py``` script to compute the panoptic score of the predictions. Make sure to have the seamless segmentation model installed.
- ```python measure_panoptic.py --gt <Location of converted WildPPS dataset> --target <Location of seamseg model predictions> --result <Location to write the results to> ```

## Publication 
If you use our dataset, model or other code please consider referencing one of our following papers

Jaus, Alexander, Kailun Yang, and Rainer Stiefelhagen. "Panoramic panoptic segmentation: Towards complete surrounding understanding via unsupervised contrastive learning." 2021 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2021. \[[PDF](https://ieeexplore.ieee.org/iel7/9575127/9575130/09575904.pdf)\]

Jaus, Alexander, Kailun Yang, and Rainer Stiefelhagen. "Panoramic panoptic segmentation: Insights into surrounding parsing for mobile agents via unsupervised contrastive learning." IEEE Transactions on Intelligent Transportation Systems (2023). \[[PDF](https://ieeexplore.ieee.org/iel7/6979/4358928/10012449.pdf)\]
