import csv
import cv2
import numpy as np

lines = []
#load images from csv file
with open('/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
#augment and preprocess images
for line in lines[1:]:
	for i in range(3):
		source_path = line[i].strip()
		current_path = '/data/'+source_path
		image = cv2.imread(current_path)
		images.append(image)
		#flip data
		images.append(cv2.flip(image,1))
		measurement = float(line[3])
		#use three cameras
		if i == 0 :
			measurements.append(measurement)
			measurements.append(measurement*-(1.0))
		elif i == 1 :
			measurements.append(measurement+0.2)
			measurements.append(measurement*-(1.0)-0.2)
		else:
			measurements.append(measurement-0.2)
			measurements.append(measurement*-(1.0)+0.2)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Cropping2D
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import tensorflow as tf

# Model from Nvidia paper
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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



#Compile and fit the model
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,batch_size=128,shuffle=True,nb_epoch=5)
model.save('/output/model.h5')