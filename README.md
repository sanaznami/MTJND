# MTJND: MULTI-TASK DEEP LEARNING FRAMEWORK FOR IMPROVED JND PREDICTION

## Introduction

This is the implementation of ... paper in Tensorflow.

**Abstract**

The limitation of the Human Visual System (HVS) in perceiving small distortions allows us to lower the bitrate required to achieve a certain visual quality. Predicting and applying the Just Noticeable Distortion (JND), which is a threshold for maximum unperceived level of distortions, is among the popular ways to do so. Recently, machine learning based methods have been able to reduce bitrate even further by improving JND prediction accuracy. However, accurate modeling of JND is very challenging, as it is highly content dependent. Furthermore, existing datasets provide little information to learn the best parameters. To remedy this issue, we propose a multi-task deep learning framework that jointly learns various complementary visual information. We design three separate methods and training strategies that jointly learn: (1) three JND levels, (2) visual attention map and a JND level, and (3) three JND levels and the visual attention map. We show that accumulating information from multiple tasks leads to a more robust prediction of JND. Experimental results confirm the superiority of our framework compared to the state-of-the-art.


**The proposed framework**

![image](https://github.com/sanaznami/MTJND/assets/59918141/5b4446af-d6bd-46aa-82db-f8266a2fe74e)

