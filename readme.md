# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./NvidiaArch.png "Nvidia Architecture"
[image2]: ./ReducedData.png "Data Histogram"

---
# Files Submitted

## 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* Behavioral-Cloning-Trainer.ipynb the jupyter notebook containing the code for the model
* drive.py for driving the car in autonomous mode(edited from original)
* model.h5 containing a trained convolution neural network 
* track1_video.mp4 a recording of the model driving on the first track
* readme.md summarizing the results(this file)

## 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around track 1 by executing 
```sh
python drive.py model.h5
```

## 3. Code

The model.py or Behavioral-Cloning-Trainer.ipynb. The files show the pipeline I used for training and validating the model, and contain comments to explain how the code works.
The drive.py file has been edited from its original code to also include preprocessing of images(lines 50-65 and 81-83). This is the same preprocessing used in the model.py which follows 
Nvidia's recomendation of changing the image size to 66x200x3 and converting it to YUV colorspace. The only difference in the preprocessing is the drive.py converts from RGB while model converts from BGR.

# Model Architecture and Training Strategy

## 1. Architecture

The model used is the suggested Nvidia architecture.
* Input size is 3@66x200
* Convolutional Layer: 36 feature maps 5x5 Kernal - Output:36@14x47
* Convolutional Layer: 48 feature maps 5x5 Kernal - Output:48@5x22
* Convolutional Layer: 64 feature maps 3x3 Kernal - Output:64@3x20
* Convolutional Layer: 64 feature maps 3x3 Kernal - Output:64@1x18
* Flatten
* Fully Connected Layer x4

Here is a visualization of the model:

![alt text][image1]

## 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 193).

## 3. Training data

Training data used was provided by Udacity.

## 4. Creation of the Training Set & Training Process

The data set initially contained 24108 images and 8036 angle recordings of track 1 from various runs and after several attempts to train the model on the data without modifications 
I realized that it contains a lot of "useless" angle values, specifically a lot(almost half) of entries where the angle is 0 which resulted in a model with a bias towards not making adjustments
and not steering as much as needed. To work around this issue I cycled through the dataset and reduced its size by removing rows of data where the same angle is repeated more than the average angles of the dataset.
The result of the data deletion can be seen in this histogram:

![alt text][image2]


This resulted in a much smaller starting training set but was overcome with the creation of extra data by mirroring the remaining images and angle values through the generator function "generate_training_data".
The validation set was created from the reduced training set through the generator function too. The generator function shuffles the data to avoid overfitting and create a reliable validation set.


#### Resources:
Training set - Udacity

Simulator - Udacity

PreProcessing Images - https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project
