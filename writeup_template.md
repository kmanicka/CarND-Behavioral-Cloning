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


#### 3. Model parameter tuning


Used Adaptive Learning Rate Optimizer, which adjusts learning rate as epochs proceeds 

```
model.compile(loss='mse', optimizer=Adadelta())
```

#### 4. Appropriate training data

Following steps were done to improve the training data. 

- The input images were normalized and cropped before the training 
- Reduce the number of samples with "Zero" Steering 
- Used Left and Right Images with corresponding changes to steering 
- Flipped the images horizontally

### Architecture and Training Documentation

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

I worked worked with the initial dataset provided with this exercies. 

Based on the data distribution I found that the number of samples with zero steering was far higher. To reduce the imact i sub-sampled such images. 

```
        train_nonzero = self.raw_data[self.raw_data.steering != 0]
        train_zero = self.raw_data[self.raw_data.steering == 0].sample(frac=.1)
        self.data = pd.concat([train_nonzero, train_zero], ignore_index=True)
```

Apart from that The data was augumented by using the left, right and flipped images. Following is the code sample. 

```
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

Based on these changes I had a set of around 4000 samples. 
Loading all the samples required lot of memory so tried using data generators.  Following link suggesed an good way to implement the same by extending keras.utils.Sequence. https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html 

Following code shows the key methods. 
```

class TrainingDataGenerator(Sequence):
    
    # initilze the class. 
    def __init__(self,base_dir) :
        self.base_dir = base_dir
        # load and prepare data  
        ....
        ....
    
    ## called by trainer after the epoch is completed 
    ## this was used to shuffel the data after each epoch. 
    def on_epoch_end(self):
        self.__prepare_data__()
    
    ## Called by trainer to get the size of the data set. 
    def __len__(self):
        return self.batches_per_epoch

    ## Called by Trainer to get a batch, 
    def __getitem__(self, index_ignore):
        return self.__get_random_transformed_batch__()
    
```

Using this approach created a Training and Validation Data Generator.
Training data shuffled and applied random transformation on the batch. 
The Validation data Generator returned the data as is. 

```
class ValidationDataGenerator(Sequence):
    
    def __init__(self, base_dir, batch_size,verbose=0) :
        self.verbose = verbose
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.data = pd.read_csv(self.base_dir + 'driving_log.csv', sep=',')
        
    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))
    
    def __getitem__(self, index):
        low = index * self.batch_size
        top = low + self.batch_size
        
        batch = self.data[low:top]
        
        filenames = batch["center"]
        X = np.array([np.array(imread(self.base_dir + fname.strip())) for fname in filenames])
        y = batch["steering"]
        
        return X,y.values
```

I was able to further speed up the training by using use_multiprocessing and 10 workers in model.fit_generator(). 


```
training_generator = TrainingDataGenerator('data/',batch_size=batch_size,batches_per_epoch=batches_per_epoch)
validation_generator = ValidationDataGenerator('data/',batch_size=batch_size)

history_object = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=epochs,
                                     verbose=1,
                                     use_multiprocessing=True, #<<<<
                                     workers=10, #<<<<
                                     callbacks=[checkpoint])
```




