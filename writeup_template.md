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

I was inspired by the VGG network and designed an architecture of a structure in which a small size convolution layer and relu and pooling are repeated.

####2. Attempts to reduce overfitting in the model

In case of Deep neural network, it is common to use Dropout, Batchnorm, etc. to reduce overfitting. However, I did not use this method.
In order to reduce the overfitting in the network architecture design stage, the following contents are reflected.

* Repeating the small filter (3x3) and activation layer reduces the risk of overfitting by increasing the complexity of the model while reducing the number of parameters.
* The channel number of the convolution layer is set as small as possible.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I collected the dataset from the first track. We collected 5 times as much as possible while trying to position the car in the middle of the road.

###Model Architecture and Training Strategy

####1. Solution Design Approach

* I designed a structure with high complexity so that training data can be learned first. I was inspired by VGGnet and designed a network of similar structure. At this stage, we found a structure where the training dataset was sufficiently learned (training error less than 0.005) without using the validation dataset.

* I then used the validation dataset to observe the degree of overfitting.
	* I used data augmentation technique only in training dataset to reduce overfitting.
	* Validation dataset does not use augmentation but only validation error.

####2. Final Model Architecture

The final model architecture (model_arch.py) is similar to vggnet, but the filter number of the convolution layer is much smaller.

Initially, the number of First Layer was increased to 32, but the number of filters in First Layer was reduced to 8 to speed training and reduce the risk of overfitting.

The following is the overall architecture.
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
* I used all the images obtained from the center camera, left camera, and right camera to increase the number of training samples.

##### 2) Create Annotation File

* I created an annotation.json file by implementing `` `create_ann_script.py```.
* The json format annotation file specifies the image filename and target angle.

```
    {
        "filename": "center_2017_08_21_20_15_44_848.jpg",
        "target": 0.05
    },
    {
        "filename": "center_2017_08_21_20_15_44_925.jpg",
        "target": 0.17258410000000002
    },
    .....
```

* I want to implement a reusable image augmentation and generator code descrived later. So, instead of using csv format log, I created a more general type of annotation file.

##### 3) Dataset Augmentation

I used the following augmentation technique.

* random shear
* random flip
* random gamma

![alt text][preprocess] 

The whole process is implemented in the `` generator / image_augment.py`` CarAugmentor class.


##### 4) Dataset Preprocessing

I performed the following pretreatment procedure to increase the training speed and reduce the risk of overfitting.

* crop
  	* I cut out the unnecessary parts from the image.
* resizing
  	* I resized the image size to (64x64).

![alt text][preprocess] 

I applied the preprocessing procedure to training set and validation set equally. Also, in order to apply the same inference process, it is implemented as a separate class from augmentation.

I have implemented the whole process in `` `generator / image_process.py`` as a Preprocessor class.







