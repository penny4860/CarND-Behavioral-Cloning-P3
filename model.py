import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle
import os

def augment_data(images, measurements):
    augment_images = []
    augment_measurements = []
    
    for img, meas in zip(images, measurements):
        augment_images.append(img)
        augment_measurements.append(meas)
        augment_images.append(cv2.flip(img, 1))
        augment_measurements.append(-1.0*meas)
    return augment_images, augment_measurements

def get_samples_in_line(line, images, measurements, image_directory):
    
    def _get_image(path):
        filename = os.path.basename(path)
        current_path = os.path.join(image_directory, filename)
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

def plot_history(history_object):
    import matplotlib.pyplot as plt
    # history_object = model.fit_generator(train_generator, samples_per_epoch =
    #     len(train_samples), validation_data = 
    #     validation_generator,
    #     nb_val_samples = len(validation_samples), 
    #     nb_epoch=5, verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    

def generator(samples, batch_size=32, do_augment=True, use_side_data=True, image_directory="dataset/images"):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                images, angles = get_samples_in_line(batch_sample, images, angles, image_directory)  # (x3)
                
            images, angles = augment_data(images, angles)                           # (x2)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# 1. Get line text from log files
log_directory = "dataset/logs"
log_files = os.listdir(log_directory)

lines = []
for log_file in log_files:
    with open(os.path.join(log_directory, log_file)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# 2. Split train/validation
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

# 3. Build Generator
train_generator = generator(train_lines, batch_size=20)
validation_generator = generator(validation_lines, batch_size=120, do_augment=False, use_side_data=False)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import MaxPooling2D, Activation, Cropping2D
   
   
    
model = Sequential()
model.add(Lambda(lambda x: (x - 128.0)/ 128.0, input_shape=(160, 320, 3)))
# (top_crop, bottom crop), (left, right)
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(8, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(BatchNormalization())
model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
   
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()
   
   
model.compile(loss='mse', optimizer='adam')
# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=2)
 
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_lines)*6,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_lines),
                                     nb_epoch=20,
                                     verbose=2)
 
plot_history(history_object)
model.save('model.h5')


