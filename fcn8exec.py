from fcn import *
from preprocess import *
from metrics import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

Epochs = 30
LearningRate = 0.0001
BatchSize = 2

FCN8model = FCN8(512)
FCN8model.summary()

chkpoint = ModelCheckpoint('roadseg.h5',monitor='val_acc',save_best_only=True,mode='auto',verbose=1)
early = EarlyStopping(monitor='val_acc',min_delta=0,patience=10,verbose=1,mode='auto',restore_best_weights=True)

opt = Adam(LearningRate)
FCN8model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',Dice_Coeff,MeanIoU(num_classes=2)])

history = FCN8model.fit(X_train,Y_train,epochs=Epochs,batch_size=BatchSize,callbacks=[chkpoint,early])

FCN8model.save('fcn8_weights.h5')

out = FCN8model.predict(X_test,batch_size=2)

predRoads = squeeze(out,axis=3)
y_test = squeeze(Y_test,axis=3)
