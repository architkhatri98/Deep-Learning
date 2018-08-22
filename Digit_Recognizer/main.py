
# coding: utf-8

# In[1]:


import pandas as pd
import keras 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#importing the datasets
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')


# In[3]:


#Viewing dataset for any NaN values
dataset.describe()


# In[4]:


dataset.info()


# In[5]:


dataset.head()


# In[6]:


#Creating Train and Test vectors
X_train = dataset.iloc[:,1:].values
y_train = dataset.iloc[:,[0]].values

X_test = testset.iloc[:,:].values


# In[7]:


#One hot encoding of y_train as it has 10 categories
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
print(y_train.shape)


# In[8]:


#Viewing shapes of our Data to confirm right shape
print(dataset.shape)
print(testset.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# In[9]:


#As the images are square, we need to convert into 28*28 size images
X_train = X_train.reshape(X_train.shape[0], 28,28)
print(X_train.shape)


# In[10]:


#Plotting random images in the training column
for i,value in enumerate(np.random.randint(0,42000, size = (4))):
    plt.subplot(2,2,i+1)
    plt.imshow(X_train[value],cmap='gray')
    plt.title(y_train[value])


# In[11]:


#As the image is in grayscale, we added 1 in the end
X_train = X_train.reshape(X_train.shape[0], 28,28,1) 
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
print(X_train.shape)
print(X_test.shape)


# In[12]:


#Creating train and validation data
from sklearn.model_selection import train_test_split
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)


# In[13]:



#Creating the CNN using Keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

classifier = Sequential()
classifier.add(Convolution2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))
classifier.add(BatchNormalization(axis = 1))
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(BatchNormalization(axis=1))
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(BatchNormalization(axis = 1))
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(BatchNormalization(axis = 1))
classifier.add(Flatten())
classifier.add(Dense(512, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(10, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[14]:


#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False, 
        zca_whitening=False,
        rotation_range=11,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=False, 
        vertical_flip=False) 


datagen.fit(X_train)


# In[15]:


#Generating results of single epoch to check the accuracy of our model
classifier.fit_generator(datagen.flow(X_train1,y_train1, batch_size=80),
                              epochs = 1, validation_data = (X_val,y_val), use_multiprocessing = True)


# In[16]:


#If the above results are good then do more epochs on complete training set
classifier.fit_generator(datagen.flow(X_train,y_train, batch_size=64),
                              epochs = 10, use_multiprocessing = True)


# In[17]:


#Predicting values using CNN
predictions = classifier.predict_classes(X_test, verbose=0)

#Creating submissions dataframe and converting into csv file
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submissions.csv", index=False, header=True)
