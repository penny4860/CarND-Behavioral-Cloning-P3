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
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(8, 5, 5,
                        border_mode='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Convolution2D(8, 5, 5,
                        border_mode='same'))
 
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()
 
 
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
plot_history(history_object)
   
model.save('model.h5')





