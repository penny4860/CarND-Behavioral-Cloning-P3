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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

���� VGG network�� ������ �޾� small size�� convolution layer�� relu, pooling�� �ݺ��Ǵ� ������ architecture�� �����Ͽ���. 


####2. Attempts to reduce overfitting in the model

Deep neural network�� ��� overfitting�� ���̱� ���� Dropout, Batchnorm���� ����ϴ� ���� �Ϲ����̴�.

�׷���, ���� �̷��� ����� ������� �ʾҴ�. 

Network architecture�� �����ϴ� �ܰ迡�� overfitting�� ���̱� ���� ������ ���� ������ �ݿ��Ͽ���. 

* small filter�� activation layer�� �ݺ��ϴ� ����� model�� complexity�� �����ϸ鼭�� parameter���ڸ� �ٿ��� overfitting�� ���輺�� ���δ�.
* convolution layer�� channel ���ڸ� ������ �۰� �����Ͽ���. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

ù ��° track���� dataset�� �����Ͽ���. �ִ��� �ڵ����� ���� ��� ��ġ�� �� �ֵ��� ����ϸ鼭 5�� �����Ͽ���.

###Model Architecture and Training Strategy

####1. Solution Design Approach

���� ���� Training data�� �н��� �� ���� ������ complexity�� ���� ������ �����Ͽ���. ���� VGGnet �� ������ �޾� ����� ������ network�� �����Ͽ���.

�� �ܰ迡���� validation dataset�� ������� �ʰ� training dataset�� ����� �н��Ǵ� (training error�� 0.005�̸�) ������ ã�Ҵ�.

���� �� �����ܰ迡�� validation dataset�� ����ؼ� overfitting ������ �����Ͽ���. ó�� �������� �������� overfitting�� ���̱� ���� parameter�� ���̴� �������� network ������ �����Ͽ���.


####2. Final Model Architecture

The final model architecture (model.py lines 34-70)�� vggnet �� ��������� convolution layer�� filter ���ڰ� �ξ� ���� �����̴�.

ó������ First Layer�� ���ڸ� 32�� ũ�� �������, Training �ӵ��� ������ �ϰ�, overfitting�� ���輺�� ���̱� ���� First Layer�� filter ���ڸ� 8�� �ٿ���.

������ ��ü���� architecture �̴�.
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

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
