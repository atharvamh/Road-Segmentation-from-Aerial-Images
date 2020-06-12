import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import asarray,expand_dims,squeeze

trainImages = sorted(glob('./dataset/vRoads_512_small/train/*.tiff'))
trainLabels = sorted(glob('./dataset/vRoads_512_small/train_labels/*.tif'))

testImages = sorted(glob('./dataset/vRoads_512_small/test/*.tiff'))
testLabels = sorted(glob('./dataset/vRoads_512_small/test_labels/*.tif'))

def plotim():
  plt.rcParams['figure.figsize'] = (10.0, 10.0)
  plt.subplots_adjust(wspace=1, hspace=0)
  count = 0
  for i in trainImages[0:4]:
    im = cv2.imread(i)
    plt.subplot(2,2,count+1)
    plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));plt.axis('off')
    count += 1
    
X1 = []
X2 = []
Y1 = []
Y2 = []

for img in trainImages[0:len(trainImages)]:
  im = cv2.imread(img)
  X1.append(im)

for img in trainLabels[0:len(trainLabels)]:
  im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
  im = asarray(im)
  im = expand_dims(im,axis=2)
  Y1.append(im)

for img in testImages[0:len(testImages)]:
  im = cv2.imread(img)
  X2.append(im)

for img in testLabels[0:len(testLabels)]:
  im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
  im = asarray(im)
  im = expand_dims(im,axis=2)
  Y2.append(im)

X = X1 + X2
X = np.array(X)
X = X/255.0

Y = Y1 + Y2
Y = np.array(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X[0:50],Y[0:50],test_size = 0.2)
