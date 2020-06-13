from keras.models import Model,load_model
import tensorflow as tf
from keras.layers.core import Dropout
from keras.layers import Input,BatchNormalization,Cropping2D,Add,add
from keras.layers.convolutional import Conv2D,UpSampling2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
import keras
from keras import layers,Sequential

def base_FCN(imsize):
  model2 = Sequential()
  model2.add(Conv2D(input_shape = (imsize,imsize,3),filters=64,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model2.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model2.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model2.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model2.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
  model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model2.add(Conv2D(4096,kernel_size=(7,7),padding='same',activation='relu',name='fc6'))

  model2.add(Conv2D(4096,kernel_size=(1,1),padding='same',activation='relu',name='fc7'))

  model2.add(Conv2D(1,kernel_size=(1,1),padding='same',activation='relu',name='score_for_classes'))

  convSize = model2.layers[-1].output_shape[2]

  model2.add(Conv2DTranspose(1,kernel_size=(4,4),strides=(2,2),padding='valid',activation=None,name='score2'))

  deconvSize = model2.layers[-1].output_shape[2]

  croppix = deconvSize - 2*convSize

  model2.add(Cropping2D(cropping=((0,croppix),(0,croppix))))

  return model2
  
def FCN8(imsize):
  fcn8 = base_FCN(imsize)
  pool4conv = Conv2D(1,kernel_size=(1,1),padding='same',activation=None,name='scorepool4')

  Summed1 = add([pool4conv(fcn8.layers[14].output),fcn8.layers[-1].output])

  x = Conv2DTranspose(2,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed1)
  x = Cropping2D(cropping=((0,2),(0,2)))(x)

  pool3conv = Conv2D(1,kernel_size=(1,1),padding='same',activation=None,name='scorepool3')

  Summed2 = add([pool3conv(fcn8.layers[10].output),x])

  Up = Conv2DTranspose(1,kernel_size=(16,16),strides = (8,8),padding = "valid",activation = None,name = "upsample")(Summed2)

  output_image = Cropping2D(cropping = ((0,8),(0,8)))(Up)

  model = Model(inputs=fcn8.input,outputs=output_image)

  return model
