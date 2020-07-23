from keras import backend as K
from keras import applications
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions 
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import sys
import cv2
import os

#Set current Model to whatever model you want to test
currentModel='Bottleneck_Norm_150_augment.h5'

print '\nEvaluating photo:',sys.argv[1],'\n'
img = image.load_img((sys.argv[1]), target_size=(150,150))

#convert image to array
img = image.img_to_array(img)
img= img/255
img = np.expand_dims(img, axis=0)

model= VGG16(weights='imagenet', include_top=False)
bottleneck_pred=model.predict(img)

model = Sequential()
model.add(Flatten(input_shape=bottleneck_pred.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#Dense must be equal to the number of classes
model.add(Dense(6, activation='softmax'))

model.load_weights(currentModel)

preds=model.predict(bottleneck_pred)
#Make sure they are in alphabetical order
names=['Amani', 'Honi', 'Kira', 'Kuchimba', 'Louis', 'Motuba']
print(preds)

index=0
count=0
frac=max(preds[0])
print(frac)
while count<5:
	count=count+1
	if preds[0][count]>preds[0][index]: index=count;


print '\nGorilla identified: ', names[index], 'with ','%.3f'%(frac*100), 'percent confidence'