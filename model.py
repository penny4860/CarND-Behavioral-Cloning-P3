import cv2
import numpy as np
import csv

def augment_data(images, measurements):
    augment_images = []
    augment_measurements = []
    
    for img, meas in zip(images, measurements):
        augment_images.append(img)
        augment_measurements.append(meas)
        augment_images.append(cv2.flip(img, 1))
        augment_measurements.append(-1.0*meas)
    return augment_images, augment_measurements

def get_samples_in_line(line, images, measurements):
    
    def _get_image(path):
        filename = path.split('/')[-1]
        current_path = "dataset/1/IMG/" + filename
        image = cv2.imread(current_path)
        return image
    
    def _get_measurement(measurement, correnction_factor):
        return measurement+correnction_factor

    
    center_img_path = line[0]
    center_meas = float(line[3])
    images.append(_get_image(center_img_path))
    measurements.append(_get_measurement(center_meas, 0))

    left_img_path = line[1]
    left_meas = float(line[4])
    images.append(_get_image(left_img_path))
    measurements.append(_get_measurement(left_meas, 0.2))

    right_img_path = line[2]
    right_meas = float(line[5])
    images.append(_get_image(right_img_path))
    measurements.append(_get_measurement(right_meas, -0.2))
    
    return images, measurements
    

lines = []
with open("dataset/1/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    images, measurements = get_samples_in_line(line, images, measurements)    

print(len(images))
images, measurements = augment_data(images, measurements)
print(len(images))
    
X_train = np.array(images)
y_train = np.array(measurements)
 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers import MaxPooling2D, Activation, Cropping2D
 
 
  
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# (top_crop, bottom crop), (left, right)
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu', border_mode='same'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu', border_mode='same'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', border_mode='same'))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
 
 
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)
    
model.save('model.h5')




