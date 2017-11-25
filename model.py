import csv
import cv2
import numpy as np

lines = []
with open('/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines[1:]:
	for i in range(3):
		source_path = line[i].strip()
		current_path = '/data/'+source_path
		image = cv2.imread(current_path)
		images.append(image)
		images.append(cv2.flip(image,1))
		measurement = float(line[3])
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

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(\
	X_train, y_train, test_size=0.2, random_state=42)

from sklearn.utils import shuffle

def generator( X, y, batch_size=32):
	num_samples = X.shape[0]
	X,y = shuffle(X,y)
	while 1:
		for offset in range(0, num_samples, batch_size):
			batch_X = X[offset:offset+batch_size]
			batch_y = y[offset:offset+batch_size]
			yield shuffle(batch_X, batch_y)

train_generator = generator(X_train, y_train, batch_size=32)
validation_generator = generator(X_validation, y_validation, batch_size=32)



from keras.models import Sequential
from keras.layers import Flatten,Dense,Cropping2D
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D


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




model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
	len(X_train), validation_data=validation_generator,\
	nb_val_samples=len(X_validation), nb_epoch=5)
model.save('/output/model.h5')