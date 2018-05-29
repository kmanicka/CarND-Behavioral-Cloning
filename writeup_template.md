# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* writeup_report.md or writeup_report.pdf summarizing the results
* modelV*.ipynb presents the various iterations of building and training the models.  modelV9.ipynb should be considered for the final submition.  
* Multiple models weights were generated and tested. weights-7C.h5 containing a best trained model
* A gif showing actual run using the best trained model. 


![Iteration 7c](https://github.com/kmanicka/CarND-Behavioral-Cloning/raw/master/iteration7c.gif)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py weights-7C.h5
```

#### 3. Submission code is usable and readable

The modelV9.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The final model architecture is a modification of a LeNet with addition of  preprocessing and dropout layers and changes for linear regressions. 

* First 2 layers are pre-processing layers used for normalizing and croping the images. 
* Next we have 3 Covnets followed by Maxpooling. 
* Then we have 3 Dense layers 
* Dropout layers are used for regularization
* we use Mean Squared Errors as the loss function 
* Adaptive Learning Rate Optimizer is used for optimization. 

#### 2. Attempts to reduce overfitting in the model

* The model contains dropout layers in order to reduce overfitting. 
* The distribution of images with "zero" Steering agenls was comparatively higher then other angles. So reduced samples with "zero" steering angles.
```
        train_nonzero = self.raw_data[self.raw_data.steering != 0]
        train_zero = self.raw_data[self.raw_data.steering == 0].sample(frac=.1)
        self.data = pd.concat([train_nonzero, train_zero], ignore_index=True)
```

#### 3. Model parameter tuning


Used Adaptive Learning Rate Optimizer, which adjusts learning rate as epochs proceeds 

```
model.compile(loss='mse', optimizer=Adadelta())
```

#### 4. Appropriate training data

Used Left, Right Images and Horizontal flipping to reduce overfitting. 


```
    def __transfomation_base(self,batch, camera_name='center', flip = False, steering_correction=1) :
        if self.verbose == 1 :
            print("__transfomation_base")
        
        filenames = batch[camera_name]
        
        X=None
        if flip == False :
            X = np.array([np.array(imread(self.base_dir + fname.strip())) for fname in filenames])
        else :
            X = np.array([np.flip(np.array(imread(self.base_dir + fname.strip())),1) for fname in filenames])
        
        y = batch["steering"] * steering_correction   
        
        return X,y.values 


    def __transfomation_center(self,batch) :
        return self.__transfomation_base(batch,'center',False,1)

    def __transfomation_left(self,batch) :
        return self.__transfomation_base(batch,'left',False,1.2)

    def __transfomation_right(self,batch) :
        return self.__transfomation_base(batch,'right',False,0.8)
        
    def __transfomation_flip(self,batch) :
        return self.__transfomation_base(batch,'center',True,-1)

    def __transfomation_left_flip(self,batch) :
        return self.__transfomation_base(batch,'left',True,-1.2)

    def __transfomation_right_flip(self,batch) :
        return self.__transfomation_base(batch,'right',True,-0.8)
```


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used an iterateve approach to come up with a model and get acceptable results. 
To start with created a workable model which could be used with drive.py. Then slowly upgraded the model and the training set to improve the results. 

Following are iterations that I used. 

* Versions 1
Setup a basic model with  appropriate input and output dimentions which can be used with drive.py
* Versions 2
Converted the model to to a Lenet Model updated for linear regression. 
* Versions 3
Normalized and cropped the imges. 
* Versions 4
Added Dropout in 3 layers with 0.25
* Versions 5
Some Code Cleanup. 
* Versions 6
Use Data Generator inspired by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html .The training became slower after using the data generator as the imges were being read 1 by 1 and processed. 
* Versions 7
Try doing bulk read of images during Data Generator, Exclude the left and right images, Implement Validation Generator, Parallel exectution of fit generator to speed things up.
Bulk read of images and parallel execution significantly improved the speed of training and training time per epoch got reduced to 10s without any impact on the model / training. 
With weights version 7C we are able to cross the sand railing.
* Versions 8
Data Augumentation do the reverse images, reverse 50% images in the batch. 
Tried different changes to the Model architecture. Used Adaptive Learning Optimizer.  
- Version 9
Reduce 0 stearing data points, Train generator will have Random Transformations and generated "defined number of batches
Validation Generator will generate all impages from Track 1 as validation samples

#### 2. Final Model Architecture

  

Following block shows the final architecture that was used. 

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_15 (Lambda)           (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_15 (Cropping2D)   (None, 96, 320, 3)        0         
_________________________________________________________________
max_pooling2d_39 (MaxPooling (None, 48, 160, 3)        0         
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 48, 160, 32)       2432      
_________________________________________________________________
activation_37 (Activation)   (None, 48, 160, 32)       0         
_________________________________________________________________
max_pooling2d_40 (MaxPooling (None, 24, 80, 32)        0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 24, 80, 64)        51264     
_________________________________________________________________
activation_38 (Activation)   (None, 24, 80, 64)        0         
_________________________________________________________________
max_pooling2d_41 (MaxPooling (None, 12, 40, 64)        0         
_________________________________________________________________
dropout_32 (Dropout)         (None, 12, 40, 64)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 12, 40, 128)       204928    
_________________________________________________________________
activation_39 (Activation)   (None, 12, 40, 128)       0         
_________________________________________________________________
max_pooling2d_42 (MaxPooling (None, 6, 20, 128)        0         
_________________________________________________________________
dropout_33 (Dropout)         (None, 6, 20, 128)        0         
_________________________________________________________________
flatten_13 (Flatten)         (None, 15360)             0         
_________________________________________________________________
dense_24 (Dense)             (None, 32)                491552    
_________________________________________________________________
activation_40 (Activation)   (None, 32)                0         
_________________________________________________________________
dropout_34 (Dropout)         (None, 32)                0         
_________________________________________________________________
dense_25 (Dense)             (None, 8)                 264       
_________________________________________________________________
activation_41 (Activation)   (None, 8)                 0         
_________________________________________________________________
dropout_35 (Dropout)         (None, 8)                 0         
_________________________________________________________________
dense_26 (Dense)             (None, 1)                 9         
=================================================================
Total params: 750,449
Trainable params: 750,449
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

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
