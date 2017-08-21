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
    break

for img, meas in zip(images, measurements):
    cv2.imshow("{}".format(meas), img)
    cv2.waitKey(0)


 
