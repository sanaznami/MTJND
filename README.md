<h1 align="center"> MTJND: MULTI-TASK DEEP LEARNING FRAMEWORK FOR IMPROVED JND PREDICTION</h1>


## Introduction

This is the implementation of MTJND: MULTI-TASK DEEP LEARNING FRAMEWORK FOR IMPROVED JND PREDICTION paper in Tensorflow.

**Abstract**

The limitation of the Human Visual System (HVS) in perceiving small distortions allows us to lower the bitrate required to achieve a certain visual quality. Predicting and applying the Just Noticeable Distortion (JND), which is a threshold for maximum unperceived level of distortions, is among the popular ways to do so. Recently, machine learning based methods have been able to reduce bitrate even further by improving JND prediction accuracy. However, accurate modeling of JND is very challenging, as it is highly content dependent. Furthermore, existing datasets provide little information to learn the best parameters. To remedy this issue, we propose a multi-task deep learning framework that jointly learns various complementary visual information. We design three separate methods and training strategies that jointly learn: (1) three JND levels, (2) visual attention map and a JND level, and (3) three JND levels and the visual attention map. We show that accumulating information from multiple tasks leads to a more robust prediction of JND. Experimental results confirm the superiority of our framework compared to the state-of-the-art.


**The proposed framework**
<p align="center">
  <img src="https://github.com/sanaznami/MTJND/assets/59918141/83777f72-da50-4087-a720-f527d6ee23e8">
</p>

<p align="center">The proposed framework and its components. (a) overall framework. (b) shared feature backbone. (c) decision tail for visual attention modeling. (d) decision tail for JND prediction. (e) MT_3LJND. (f) MT_1LJND_VA. (g) MT_3LJND_VA</p>


## Requirements

- Tensorflow
- FFmpeg


## Dataset

Our evaluation is conducted on [VideoSet](https://ieee-dataport.org/documents/videoset) and [MCL-JCI](https://mcl.usc.edu/mcl-jci-dataset/) datasets.


## Pre-trained Models
Our pre-trained models can be downloaded using this .... .


## Usage
Our pretrained models are capable of predicting JND values, and they can also be employed for training on a custom dataset.
##### Note: The dataset used for training and testing should have such a structure.

    - rootdir/
         - train/
             - img#1
             - img#2
             - ...
             - JND-Levels.txt (a file containing the 3 JND levels per image)
         - valid/
             - img#1
             - img#2
             - ...
             - JND-Levels.txt (a file containing the 3 JND levels per image)
         - test/
             - img#1
             - img#2


##### MT_3LJND
For prediction, the following commands can be used.

    python3 MT_3LJND.py test --data_dir "Path to the folder containing train, valid, and test subfolders/" --model_weights_path "Path to the model weights file for testing" --result_path "Path to save test results"

For training, the following commands can be used.

    python3 MT_3LJND.py train --data_dir "Path to the folder containing train, valid, and test subfolders/" --checkpoint_path "Path to save checkpoints" --csv_log_path "Path to save CSV logs during training" --epochs Number-of-training-epochs --batch_size Batch-size-for-training --learning_rate Learning-rate-for-optimizer

   

### Testing

### Training


## Citation

If our work is useful for your research, please cite our paper:

    @inproceedings{nami2023mtjnd,
    	title={MTJND: MULTI-TASK DEEP LEARNING FRAMEWORK FOR IMPROVED JND PREDICTION},
	author={Nami, Sanaz and Pakdaman, Farhad and Hashemi, Mahmoud Reza and Shirmohammadi, Shervin and Gabbouj, Moncef},
	booktitle={Proceedings of the IEEE International Conference on Image Processing (ICIP)},
	year={2023}
    }


## Contact

If you have any question, leave a message here or contact Sanaz Nami (snami@ut.ac.ir, sanaz.nami@tuni.fi).


