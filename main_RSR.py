# -*- coding: utf-8 -*-
"""
--------------- ROAD SIGN RECOGNITION ---------------
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import csv

import skimage.morphology as morph
from skimage.filters import rank
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import h5py

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

from keras import backend as K
K.set_image_dim_ordering('tf')
# K.set_image_dim_ordering('th')

"""
import keras
config = tf.ConfigProto(device_count = {'GPU': 1, 'CPU': 8}) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
"""

# --- Loading data

trainingFile, validationFile, testingFile = "./traffic_sign_data/train.p", "./traffic_sign_data/valid.p", "./traffic_sign_data/test.p"

with open(trainingFile, mode='rb') as file:
    train = pickle.load(file)
with open(validationFile, mode='rb') as file:
    valid = pickle.load(file)
with open(testingFile, mode='rb') as file:
    test = pickle.load(file)

# --- Mapping traffic sign names and ClassID
signs = []
with open('signnames.csv','r') as csvfile:
    signNames = csv.reader(csvfile, delimiter=',')
    next(signNames,None)
    for row in signNames:
        signs.append(row[1])
    csvfile.close()

# --- Dataset exploration
    
XTrain, yTrain = train['features'], train['labels']
XValid, yValid = valid['features'], valid['labels']
XTest, yTest = test['features'], test['labels']

trainLength, testLength, validationLength = XTrain.shape[0], XTest.shape[0], XValid.shape[0] # numer of examples
IMG_SHAPE = XTrain[0].shape # shape of an image
NUM_CLASSES = len(np.unique(yTrain)) # number of unique labels in the dataset

""" _DOC_RP_ : dataset_dimensions
print("Number of training examples: ", trainLength)
print("Number of testing examples: ", testLength)
print("Number of validation examples: ", validationLength)
print("Image data shape =", IMG_SHAPE)
print("Number of classes =", NUM_CLASSES)
"""

def displayImages(dataset, dataset_y, ylabel="", cmap=None):
    # Displays a list of images
    plt.figure(figsize=(15,16))
    
    for k in range(6):
        plt.subplot(1, 6, k+1)
        idx = random.randint(0, len(dataset))
        # We're using grayscale if there's only one color channel
        cmap = 'gray' if len(dataset[idx].shape) == 2 else cmap
        plt.imshow(dataset[idx], cmap=cmap)
        plt.xlabel(signs[dataset_y[idx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

""" _DOC_RP_ : dataset_samples
displayImages(XTrain, yTrain, "Training")
displayImages(XTest, yTest, "Test")
displayImages(XValid, yValid, "Validation")
"""

def histogramPlot(dataset, label):
    # Plots a histogram of the input data
    hist, bins = np.histogram(dataset, bins=NUM_CLASSES)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()

""" _DOC_RP_ : dataset_histogram
histogramPlot(yTrain, "Training")
histogramPlot(yTest, "Test")
histogramPlot(yValid, "Validation")
"""

# --- Preprocessing (shuffling, grayscaling, contrasting, normalization)

# Shuffling
XTrain, yTrain = shuffle(XTrain, yTrain)

# Grayscaling
def GrayScale(img):
    return(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

""" _DOC_RP_ : gray_pictures """
# grayImages = list(map(GrayScale, XTrain))
# displayImages(grayImages, yTrain, "Grayscaled images", "gray")

# Contrasting
def LocalHistogramEqualize(img):
    # Uses local histogram equalization on grayscale images
    kernel = morph.disk(30)
    imgLocal = rank.equalize(img, selem=kernel)
    return(imgLocal)
  
""" _DOC_RP_ : equalized_pictures """
# equalizedImages = list(map(LocalHistogramEqualize, grayImages))
# displayImages(equalizedImages, yTrain, "Equalized images", "gray")

# Normalization
def NormalizeImage(img):
    # Normalizes images to [0,1] scale
    return(np.divide(img,255))

""" _DOC_RP_ : normalized_pictures """
# normalizedImages = np.zeros((XTrain.shape[0], XTrain.shape[1], XTrain.shape[2]))

# for i, img in enumerate(equalizedImages):
  #   normalizedImages[i] = NormalizeImage(img)
    
# displayImages(normalizedImages, yTrain, "Normalized images", "gray")
# normalizedImages = normalizedImages[...,None]

# Final preprocessing pipeline
def Preprocess(data):
    grayImages = list(map(GrayScale, data))
    equalizedImages = list(map(LocalHistogramEqualize, grayImages))
    n = data.shape
    normalizedImages = np.zeros((n[0],n[1],n[2]))
    for i, img in enumerate(equalizedImages):
        normalizedImages[i] = NormalizeImage(img)
    normalizedImages = normalizedImages[...,None]
    return(normalizedImages)


# --- Model creation (more info about convnets, optimizers, momentum... in RP) using Keras on Tensorflow backend

def CNNModel():
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SHAPE[0],IMG_SHAPE[1],1))) # (32,32,1) after preprocessing
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return(model)
    
modelSGD = CNNModel()
modelADAM = CNNModel()

# Training our first two models using SGD and ADAM
learningRate = 0.01
sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=learningRate, decay=1e-6)

modelSGD.compile(loss='sparse_categorical_crossentropy',
                 optimizer=sgd,
                 metrics=['accuracy'])
modelADAM.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

def LRSchedule(epoch):
    return learningRate*(0.1**int(epoch/10))

tensorboard = TensorBoard(log_dir='logs/{}', write_graph=True, write_images=True)

# --- Model training (SGD = stochastic gradient descent)

batchSize, nbEpoch = 32, 20

# Plotting training (acc vs. epoch) (ex. SGD)
"""
history = modelSGD.fit(Preprocess(XTrain), yTrain,
             batch_size=batchSize,
             epochs=nbEpoch,
             validation_split=0.2,
             shuffle=True,
             callbacks=[LearningRateScheduler(LRSchedule),
                        ModelCheckpoint('modelSGD.h5',save_best_only=True)]
             )

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
"""
"""
modelSGD.fit(Preprocess(XTrain), yTrain,
             batch_size=batchSize,
             epochs=nbEpoch,
             validation_split=0.2,
             shuffle=True,
             callbacks=[LearningRateScheduler(LRSchedule),
                        ModelCheckpoint('modelSGD.h5',save_best_only=True),
                        tensorboard]
             )

# --- Model architecture display & testing on validation & test data
"""
modelSGD = load_model('modelSGD.h5')
modelSGD.summary()

# validLoss, validAcc = modelSGD.evaluate(Preprocess(XValid), yValid)
# print('Validation accuracy: ', validAcc)

# testLoss, testAcc = modelSGD.evaluate(Preprocess(XTest), yTest)
# print('Test accuracy: ', testAcc)

# --- Model predicting on test data
# print(modelSGD.predict(Preprocess(XTest)))


# --- More infos on the model (use .get_weights() to get weights & biases)