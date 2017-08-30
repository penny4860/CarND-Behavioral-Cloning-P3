#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[preprocess]: ./examples/preprocess.png "preprocess"
[augment]: ./examples/augment.png "augment"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* generator package containing data augmentation and data generating code
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained parameters 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

나는 VGG network에 영감을 받아 small size의 convolution layer와 relu, pooling이 반복되는 구조의 architecture를 설계하였다. 


####2. Attempts to reduce overfitting in the model

Deep neural network의 경우 overfitting을 줄이기 위해 Dropout, Batchnorm등을 사용하는 것이 일반적이다.

그러나, 나는 이러한 방식을 사용하지 않았다. 

Network architecture를 설계하는 단계에서 overfitting을 줄이기 위해 다음과 같은 내용은 반영하였다. 

* small filter(3x3)와 activation layer를 반복하는 방식은 model의 complexity를 높게하면서도 parameter숫자를 줄여서 overfitting의 위험성을 줄인다.
* convolution layer의 channel 숫자를 가급적 작게 설정하였다.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

나는 첫 번째 track에서 dataset을 수집하였다. 최대한 자동차가 길의 가운데 위치할 수 있도록 노력하면서 5번 수집하였다.

###Model Architecture and Training Strategy

####1. Solution Design Approach

* 나는 먼저 Training data가 학습될 수 있을 정도로 complexity가 높은 구조를 설계하였다. 나는 VGGnet 에 영감을 받아 비슷한 구조의 network를 설계하였다. 이 단계에서는 validation dataset은 사용하지 않고 training dataset이 충분히 학습되는 (training error가 0.005미만) 구조를 찾았다.

* 나는 그 다음단계에서 validation dataset을 사용해서 overfitting 정도를 관찰하였다. 
	* 나는 overfitting을 줄이기 위해 training dataset에서만 data augmentation 기법을 사용하였다.
	* validation dataset에서는 augmentation을 사용하지 않고 validation error만을 관찰하였다.

####2. Final Model Architecture

The final model architecture (model.py lines 34-70)는 vggnet 과 비슷하지만 convolution layer의 filter 숫자가 훨씬 작은 구조이다.

처음에는 First Layer의 숫자를 32로 크게 잡았으나, Training 속도를 빠르게 하고, overfitting의 위험성을 줄이기 위해 First Layer의 filter 숫자를 8로 줄였다.

다음은 전체적인 architecture 이다.
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 8)     224         lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 64, 64, 8)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 64, 8)     584         activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 64, 64, 8)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 32, 8)     0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 32, 32, 16)    1168        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 32, 32, 16)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 32, 32, 16)    2320        activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 32, 32, 16)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 16, 16, 16)    0           activation_4[0][0]               
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 16, 16, 32)    4640        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 16, 16, 32)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 16, 16, 32)    9248        activation_5[0][0]               
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 16, 16, 32)    0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 8, 8, 32)      0           activation_6[0][0]               
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 8, 8, 64)      18496       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 8, 8, 64)      0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 8, 8, 64)      36928       activation_7[0][0]               
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 8, 8, 64)      0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 4, 4, 64)      0           activation_8[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1024)          0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           524800      flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           65664       activation_9[0][0]               
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 128)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             129         activation_10[0][0]              
====================================================================================================
```

####3. Creation of the Training Set & Training Process

##### 1) Training Set Collection

* To capture good driving behavior, I recorded five laps on track one using center lane driving. 
* Training sample 숫자를 늘리기 위해 center camera, left camera, right camera에서 획득한 image를 모두 사용하였다. 

##### 2) Create Annotation File

* 나는 ```create_ann_script.py```를 구현해서 annotation.json file을 만들었다.
* json format의 annotation file에는 image filename과 target angle이 명시되어 있다.

```
    {
        "filename": "center_2017_08_21_20_15_44_772.jpg",
        "target": 0.0
    },
    {
        "filename": "center_2017_08_21_20_15_44_848.jpg",
        "target": 0.05
    },
    {
        "filename": "center_2017_08_21_20_15_44_925.jpg",
        "target": 0.17258410000000002
    },
```

##### 3) Dataset Augmentation

나는 다음과 같은 augmentation 기법을 사용하였다. 

* random shear
* random flip
* random gamma

![alt text][preprocess] 

모든 과정은 ```generator/image_augment.py``` 의 CarAugmentor class에 구현되어있다. 


##### 4) Dataset Preprocessing

나는 Training 속도를 높이고 overfitting을 위험성을 줄이기 위해 다음과 같은 전처리 과정을 수행하였다.

* crop
  	* image에서 불필요한 부분을 잘라내었다.
* resizing
  	* image size를 (64x64)로 변환하였다.

![alt text][preprocess] 

나는 preprocessing 과정을 training set과 validation set에 동일하게 적용하였다. 또한 inference 과정에서도 동일하게 적용하기 위해서 augmentation 과는 별도의 class 로 구현하였다. 

```generator/image_process.py``` 에 Preprocessor class로 모든 과정을 구현하였다.







