
# coding: utf-8

# In[1]:


import sklearn
import csv
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
#from random import shuffle
from os import getcwd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.utils import shuffle
print("done")


# ## Preprocess image function (distortion, color space conversion)

# In[2]:


def preprocess_image(img):
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

print("Done")


# ## Distort the images function

# In[ ]:


def random_distort(img, angle):
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)
print("done")


# ## Generate the training data with pre processing

# In[3]:


def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = preprocess_image(img)
            #if not validation_flag:
            #    img, angle = random_distort(img, angle)
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)
            # flip horizontally and invert steer angle, if magnitude is > 0.33 to avoid adding 
            # too much data without meaningful change
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)
print("done")


# ## Load the data from the CSV

# In[4]:


lines=[]
image_paths = []
angles = []
img_path_prepend = ['', getcwd() + '/data/']
with open('./data/driving_log.csv') as csvfile:
    driving_data = list(csv.reader(csvfile, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    for row in driving_data[1:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue
        # get center image path and angle
        image_paths.append(img_path_prepend[1] + row[0])
        angles.append(float(row[3]))
        # get left image path and angle
        image_paths.append(img_path_prepend[1] + row[1])
        #add a correction factor of .25 to the angle
        angles.append(float(row[3])+0.25)
        # get left image path and angle
        image_paths.append(img_path_prepend[1] + row[2])
        # add a correction factor of -.25 to the angle
        angles.append(float(row[3])-0.25)

image_paths = np.array(image_paths)
angles = np.array(angles)
print('Size of data:', image_paths.shape, angles.shape)


# ## Display the data distribution and remove unwanted amount of driving without turns

# In[5]:


num_bins = 23
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')

keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

print('Size of data after removing unwanted values:', image_paths.shape, angles.shape)


# ## Split the datasets

# In[6]:


image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                  test_size=0.05, random_state=42)

print('Train:', image_paths_train.shape, angles_train.shape)
print('Test:', image_paths_test.shape, angles_test.shape)


# ## Keras model with Nvidia CNN

# In[7]:


model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(66,200,3)))
##Nvidia Model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
val_gen = generate_training_data(image_paths_train, angles_train, validation_flag=True, batch_size=64)
test_gen = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)
##save the model after every epoch
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')    

history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040, 
                                  nb_epoch=5, verbose=2, callbacks=[checkpoint])

model.save('model.h5')
print("done")


# In[ ]:




