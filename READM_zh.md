# RoadDamageDetector

********
# [Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022)(基于群体感知的道路损坏检测挑战)](https://crddc2022.sekilab.global/) 

## 出版物
- 获奖者解决方案详细回顾(2024) - [从全球挑战到本地解决方案：道路损坏检测中跨国合作和获胜策略的回顾](https://doi.org/10.1016/j.aei.2024.102388)
- 摘要论文（获奖者、任务、程序） - [基于群体感知的道路损坏检测挑战(CRDDC’2022)](https://www.researchgate.net/publication/367456896_Crowdsensing-based_Road_Damage_Detection_Challenge_CRDDC'2022)
- 数据文章[RDD2022: 用于自动道路病害检测的跨国影像数据集](https://www.researchgate.net/publication/363668453_RDD2022_A_multi-national_image_dataset_for_automatic_Road_Damage_Detection)

## 数据集
- 通过CRDDC'2022发布的详细统计数据和其他信息可以在篇[文章](https://www.researchgate.net/publication/363668453_RDD2022_A_multi-national_image_dataset_for_automatic_Road_Damage_Detection)里找到。

- 通过CRDDC发布的RDD2022数据集现在也可以在[FigShare Repository](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)上找到！如果你使用这些数据或信息，请标注引用信息。

- [RDD2022.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip)
  - `RDD2022.zip`包含了来自六个国家的训练和测试数据：日本、印度、捷克共和国、挪威、美国和中国。
  - 训练集提供了图片(.jpg)和标注(.xml)，注释的格式和pascalVOC相同。
  - 测试集只提供了图片。

- 与RDD2020数据和CRDDC提交相关的补充文件：
    -  [Directory_Structure_CRDDC_RDD2022.txt](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Directory_Structure_CRDDC_RDD2022.txt)

    - [File_List_CRDDC_RDD2022.txt](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/File_List_CRDDC_RDD2022.txt)

    - [label_map.pbtxt](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/label_map.pbtxt)

    - [sampleSubmission_covering_India_Japan_and_Czech.txt](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/sampleSubmission.txt)

- 下载指定国家数据集的链接：
    - [RDD2022_Japan.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip) (1022.9 MB - train and test)
    - [RDD2022_India.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip) (502.3 MB - train and test)
    - [RDD2022_Czech.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip)  (245.2 MB - train and test)
    - [RDD2022_Norway.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Norway.zip)  (9.9 GB - train and test)
    - [RDD2022_United_States.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip)  (423.8 MB - train and test)
    - [RDD2022_China_MotorBike.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip)  (183.1 MB - train and test)
    - [RDD2022_China_Drone.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_Drone.zip)  (152.8 MB - only train)
 
## 需要考虑的道路损坏类型
```
{
  D00: Longitudinal Crack 纵向裂缝, 
  D10: Transverse Crack 横向裂缝, 
  D20: Aligator Crack 鳄鱼裂缝, 
  D40: Pothole 坑洼
}
```

## 引用

```csv
@article{2024_ARYA_CRDDC_review,
title = {From global challenges to local solutions: A review of cross-country collaborations and winning strategies in road damage detection},
author = {Deeksha Arya and Hiroya Maeda and Yoshihide Sekimoto},
journal = {Advanced Engineering Informatics},
volume = {60},
pages = {102388},
year = {2024},
doi = {https://doi.org/10.1016/j.aei.2024.102388},
}

@inproceedings{arya2022crowdsensing,
  title={Crowdsensing-based Road Damage Detection Challenge (CRDDC’2022)},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Omata, Hiroshi and Kashiyama, Takehiro and Sekimoto, Yoshihide},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={6378--6386},
  year={2022},
  organization={IEEE}
}

@article{arya2022rdd2022,
  title={RDD2022: A multi-national image dataset for automatic Road Damage Detection},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal={arXiv preprint arXiv:2209.08538},
  year={2022}
}

@article{arya2021deep,
  title={Deep learning-based road damage detection and classification for multiple countries},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Mraz, Alexander and Kashiyama, Takehiro and Sekimoto, Yoshihide},
  journal={Automation in Construction},
  volume={132},
  pages={103935},
  year={2021},
  publisher={Elsevier}
}

@article{arya2021rdd2020,
  title={RDD2020: An annotated image dataset for automatic road damage detection using deep learning},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal={Data in brief},
  volume={36},
  pages={107133},
  year={2021},
  publisher={Elsevier}

@inproceedings{arya2020global,
  title={Global road damage detection: State-of-the-art solutions},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Omata, Hiroshi and Kashiyama, Takehiro and Sekimoto, Yoshihide},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},
  pages={5533--5539},
  year={2020},
  organization={IEEE}
}
```

********

# [Global Road Damage Detection Challenge 全球道路损坏检测挑战(GRDDC'2020)](https://rdd2020.sekilab.global/) 

## 视频
查看关于GRDDC'2020（美国乔治亚洲亚特兰大）的详细视频！

[![介绍视频](https://img.youtube.com/vi/8sh70wjn1aI/0.jpg)](https://youtu.be/8sh70wjn1aI "Introduction Video")

## 出版物


The details of the Global Road Damage Detection Challenge (GRDDC) 2020, held as an IEEE Big Data Cup with a worldwide participation of 121 teams, are encapsulated in the paper [Global Road Damage Detection: State-of-the-art Solutions](https://www.researchgate.net/publication/350199109_Global_Road_Damage_Detection_State-of-the-art_Solutions). 

**Citation:** 
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., & Sekimoto, Y. (2020). Global Road Damage Detection: State-of-the-art Solutions. IEEE International Conference on Big Data (Big Data), Atlanta, GA, USA, 2020, pp. 5533-5539, doi: 10.1109/BigData50022.2020.9377790.

Follow the [project](https://www.researchgate.net/project/Global-Road-Damage-Detection) for further updates on the publications!

## Dataset for GRDDC 2020
- [train.tar.gz](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/train.tar.gz)
  - `train.tar.gz` contains Japan/India/Czech images and annotations. The format of annotations is the same as pascalVOC.

- [test1.tar.gz](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/test1.tar.gz)

- [sampleSubmission.txt](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/sampleSubmission.txt)

- [test2.tar.gz](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/IEEE_bigdata_RDD2020/test2.tar.gz)

## Citation for the GRDDC (InJaCz) Dataset
The data collection methodology, study area and other information for the India-Japan-Czech dataset are provided in our research papers entitled [Deep learning-based road damage detection and classification for multiple countries](https://doi.org/10.1016/j.autcon.2021.103935), and 
[RDD2020: An annotated image dataset for Automatic Road Damage Detection using Deep Learning](https://www.sciencedirect.com/science/article/pii/S2352340921004170)!

The dataset utilizes the RDD-2019 data introduced in [Generative adversarial network for road damage detection](https://www.researchgate.net/publication/341836638_Generative_adversarial_network_for_road_damage_detection).

If you use or find our dataset and/or article useful, please cite the following:

1. **[Latest Research Article](https://doi.org/10.1016/j.autcon.2021.103935):** Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2021). Deep learning-based road damage detection and classification for multiple countries. Automation in Construction, 132, 103935. 10.1016/j.autcon.2021.103935.
2. **[RDD-2020 Data Article](https://www.sciencedirect.com/science/article/pii/S2352340921004170):** Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2021). RDD2020: An annotated image dataset for automatic road damage detection using deep learning. Data in brief, 36, 107133. 10.1016/j.dib.2021.107133.
3. **[RDD-2019 Article](https://www.researchgate.net/publication/341836638_Generative_adversarial_network_for_road_damage_detection):** Maeda, H., Kashiyama, T., Sekimoto, Y., Seto, T. and Omata, H. (2020). Generative adversarial network for road damage detection. Computer‐Aided Civil and Infrastructure Engineering, 36(1), pp.47-60.
4. **[GRDDC Summary Paper](https://www.researchgate.net/publication/350199109_Global_Road_Damage_Detection_State-of-the-art_Solutions?ev=project):** Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., & Sekimoto, Y. (2020). Global Road Damage Detection: State-of-the-art Solutions. IEEE International Conference on Big Data (Big Data), Atlanta, GA, USA, 2020, pp. 5533-5539, doi: 10.1109/BigData50022.2020.9377790.
___________________________________________________
[**dataset**] Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., Seto, T., Mraz, A., & Sekimoto, Y. (2021), “RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification”, Mendeley Data, V1, doi: 10.17632/5ty2wb6gvg.1
____________________________________________________________
**arXiv Pre-print**: 
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2020). Transfer Learning-based Road Damage Detection for Multiple Countries. arXiv preprint arXiv:2008.13101.

## Damage Categories to be considered
{D00: Longitudinal Crack, D10: Transverse Crack, D20: Aligator Crack, D40: Pothole}

****************************

# Road Damage Dataset 2019

## Citation

If you use or find out our dataset useful, please cite [our paper](https://doi.org/10.1111/mice.12561) in the journal of [Computer-Aided Civil and Infrastructure Engineering](https://onlinelibrary.wiley.com/journal/14678667):

Maeda, H., Kashiyama, T., Sekimoto, Y., Seto, T. and Omata, H. (2020). Generative adversarial network for road damage detection. Computer‐Aided Civil and Infrastructure Engineering, 36(1), pp.47-60.


## Abstract
Machine learning can produce promising results when sufficient training data are available; however, infrastructure inspections typically do not provide sufficient training data for road damage. Given the differences in the environment, the type of road damage and the degree of its progress can vary from structure to structure. The use of generative models, such as a generative adversarial network (GAN) or a variational autoencoder, makes it possible to generate a pseudoimage that cannot be distinguished from a real one. Combining a progressive growing GAN along with Poisson blending artificially generates road damage images that can be used as new training data to improve the accuracy of road damage detection. The addition of a synthesized road damage image to the training data improves the F‐measure by 5% and 2% when the number of original images is small and relatively large, respectively. All of the results and the new Road Damage Dataset 2019 are publicly available. 


## The structure of Road Damage Dataset 
The structure of the Road Damage Dataset 2019 is the same as the previous one: Pascal VOC.
 
## Download Road Damage Dataset
Please pay attention to the disk capacity when downloading.
- trainedModels
  - [Resnet(128MB)](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/Japan/CACAIE2020/frozen_inference_graph_resnet.pb)
  - [Mobilenet(18MB)](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/Japan/CACAIE2020/frozen_inference_graph_mobilenet.pb)

- [RoadDamageDataset_2019 (2.4GB)](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/Japan/RDD2020_data.tar.gz)

********

# Road Damage Dataset 2018

## Citation

If you use or find out our dataset useful, please cite [our paper](https://doi.org/10.1111/mice.12387) in the journal of [Computer-Aided Civil and Infrastructure Engineering](https://onlinelibrary.wiley.com/journal/14678667):

Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. 
Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images. 
Computer‐Aided Civil and Infrastructure Engineering.

@article{maedaroad, title={Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images}, 
author={Maeda, Hiroya and Sekimoto, Yoshihide and Seto, Toshikazu and Kashiyama, Takehiro and Omata, Hiroshi}, 
journal={Computer-Aided Civil and Infrastructure Engineering}, publisher={Wiley Online Library} }

arXiv version is [here](https://arxiv.org/abs/1801.09454).


## Abstract

Research on damage detection of road surfaces using image processing techniques has been actively conducted achieving considerably high detection accuracies.
However, many studies only focus on the detection of the presence or absence of damage. However, in a real-world scenario, when the road managers from a governing body needs to repair such damage, they need to know the type of damage clearly to take effective action. In addition, in many of these previous studies, the researchers acquire their own data using different methods. Hence, there is no uniform road damage dataset available openly, leading to the absence of a benchmark for road damage detection.
This study makes three contributions to address these issues.
First, to the best of our knowledge, for the first time, a large-scale road damage dataset is prepared. This dataset is composed of 9,053 road damage images captured with a smartphone installed on a car, with 15,435 instances of road surface damage included in these road images. These images are captured in a wide variety of weather and illuminance conditions. In each image, the bounding box representing the location of the damage and the type of damage are annotated.
Next, we use the state-of-the-art object detection method using convolutional neural networks to train the damage detection model with our dataset, and compare the accuracy and runtime speed on both, a GPU server and a smartphone. Finally, we show that the type of damage can be classified into eight types with high accuracy by applying the proposed object detection method.
The road damage dataset, our experimental results, and the developed smartphone application used in this study are made publicly available.
This page introduces the road damage dataset we created.


## The structure of Road Damage Dataset 
Road Damage Dataset contains trained models and Annotated images.
Annotated images are presented as the same format to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
- trainedModels
    - SSD Inception V2
    - SSD MobileNet
- RoadDamageDataset (dataset structure is the same format as PASCAL VOC)
    - Adachi
        - JPEGImages : contains images
        - Annotations : contains xml files of annotation
        - ImageSets : contains text files that show training or evaluation image list
    - Chiba
    - Muroran
    - Ichihara
    - Sumida
    - Nagakute
    - Numazu

## Download Road Damage Dataset
Please pay attention to the disk capacity when downloading.
- [trainedModels (70MB)](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/Japan/CACAIE2018/trainedModels.tar.gz)

- [RoadDamageDataset_v1 (1.7GB)](https://mycityreport.s3-ap-northeast-1.amazonaws.com/02_RoadDamageDataset/public_data/Japan/CACAIE2018/RoadDamageDataset.tar.gz)

## Dataset Tutorial

我们也创建了一个教程来展示如何使用道路损坏数据集。   
教材包括以下内容：
- 如何下载道路损坏数据集
- 数据集的结构
- 数据集的统计信息
- 如何使用训练好的模型

请查看[RoadDamageDatasetTutorial.ipynb](https://github.com/sekilab/RoadDamageDetector/blob/master/RoadDamageDatasetTutorial.ipynb).

********

# Privacy matters
Our dataset is openly accessible by the public. Therefore, considering issues with privacy, based on visual inspection, when a person's face or a car license plate are clearly reflected in the image, they are blurred out.

# License
Images on this dataset are available under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0). The license and link to the legal document can be found next to every image on the service in the image information panel and contains the CC BY-SA 4.0 mark:
<br><a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/deed.en"><img alt="Creative Commons License" style="border-width:0" src="https://licensebuttons.net/l/by-sa/4.0/88x31.png" /></a><br />

