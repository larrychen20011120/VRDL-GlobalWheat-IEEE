# VRDL Final Project (Global Wheat 2020)


## Project Introduction

Wheat represents a fundamental staple crop globally, with its cultivation and quality constituting critical concerns for agricultural producers, governmental bodies, and consumers alike. In recent years, there has been a marked increase in scholarly and practical interest in the application of computer vision and deep learning methodologies for the detection and analysis of wheat crops. In this study, we propose a one-stage approach for global wheat detection that leverages **YOLOv8**, a cutting-edge object detection algorithm, in conjunction with **pseudo-labeling** techniques and the incorporation of **external data sources**. This framework is designed to enhance the accuracy and efficiency of wheat identification in images, and we provide a comprehensive explanation of its implementation and potential benefits. Our experiments yielded a breakthrough: **we surpassed the previous first-place performance by an impressive 3.4% in mAP[50:75]**, demonstrating the superior capabilities of our method. 

## Set up the Environment

- download dependencies
    ```
    conda env create --file environment.yml
    ```
- activate env
    ```
    conda acctivate DL-Image
    ```

## Run Our Codes

### Our Methods

#### Prepared the Dataset

We extend 2020 dataset with more diverse wheat data from 2021 dataset. It represents 4 new countries, 22 new measurements sessions.

-  Download the dataset from [kaggle competition 2020](https://www.kaggle.com/competitions/global-wheat-detection/data)

- Download the [2021 dataset](https://www.kaggle.com/datasets/bendvd/global-wheat-challenge-2021) from someone released in Kaggle

- Download the [2020 testing set](https://www.global-wheat.com/gwhd.html) from Global Wheat Platform

- Ensure your project is like the following tree structure, where `gwhd_2020`, `gwhd_2021`, `GlobalWheatTestSet` are directories representing the competition dataset of 2020, 2021 and unlabeled testing set of 2020, respectively.
    ```
    VRDL-GlobalWheat-IEEE
    ├── environment.yml
    ├── GlobalWheatTestSet
    │   ├── nau_1/
    │   ├── uq_1/
    │   ├── utokyo_1/
    │   └── utokyo_2/
    ├── gwhd_2020
    │   ├── train/
    │   ├── test/
    │   ├── train.csv
    │   └── sample_submission.csv
    ├── gwhd_2021
    │   ├── competition_test.csv
    │   ├── competition_train.csv
    │   ├── competition_val.csv
    │   ├── images/
    │   └── metadata_dataset.csv
    ├── pl.yaml
    ├── README.md
    ├── *.py
    ├── tools/
    ```

- Run the following scripts to transform the dataset format to YOLO format
    ```
    python prepare_dataset.py   
    python prepare_dataset_2021.py
    ```
    > You will see two directories are created: `datasets/2020`, `datasets/2021` and 2 yaml file is generated `wheat.yaml`, `wheat_2021.yaml`

#### Finetune the YOLOv8 Model
We use **Ultralytics** framework to finetune the YOLOv8 model. The detailed hyper-parameter settings is shown in the following table.

- For training YOLOv8 on 2020 dataset
```
python train.py
```
- For training YOLOv8 on 2021 dataset
```
python train2021.py
```

#### Pseudo Labeling
In the 2021 dataset paper, it demonstrates that the test dataset was biased in comparison to the training dataset. Hence, we apply the semi-supervised learning method, pseudo labeling, to ultilize the high confidence testing set to improve our performance.

- Use the best model from finetuning to retrain with pseudo labels:
```
python pseudo_labeling.py
```

### Find Best Confidence Threshold

A helper script is included to scan different confidence thresholds on your validation set and find the one that yields the highest mAP@[0.5:0.75].

1. **Run the script**:
    ```bash
    python tools/find_best_confidence.py
    ```
    This will:
    - Load `VDL_best.pt`
    - Evaluate on the validation split at image size 1024, batch size 36, using 18 workers
    - Sweep confidence from 0.10 to 0.30 in 20 steps
    - Print out AP@[0.5:0.75] for each confidence and report the best threshold at the end.

2. **(Optional)** If you want to adjust the range, step count, or data path, edit the `conf_list` or file paths at the top of `tools/find_best_confidence.py`.


### Reproduce of the Top-Rank Methods


## Reference and Technical Supports
1. [Global Wheat Head Dataset 2021: more diversity to improve the benchmarking of wheat head localization methods](https://arxiv.org/abs/2105.07660)
2. [Ultralytics Library](https://github.com/ultralytics/ultralytics)