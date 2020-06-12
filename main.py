from preprocess import *
from UNet import *
from metrics import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

Epochs = 10
LearningRate = 0.0001
BatchSize = 5

if __name__ == "__main__":
  unet()
  chkpoint = ModelCheckpoint('roadseg.h5',monitor='val_acc',save_best_only=True,mode='auto',verbose=1)
  early = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=1,mode='auto',restore_best_weights=True)

  opt = Adam(LearningRate)
  model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',Dice_Coeff,MeanIoU(num_classes=2)])

  history = model.fit(X_train,Y_train,epochs=Epochs,batch_size=BatchSize,callbacks=[chkpoint,early])
  
  out = model.predict(X_test,batch_size=5)
  
  predRoads = squeeze(out,axis=3)
  y_test = squeeze(Y_test,axis=3)
