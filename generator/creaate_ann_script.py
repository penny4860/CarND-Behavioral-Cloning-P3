# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os

LOG_FILE = './data/driving_log.csv'
ANNOTATION_FILE = 'annotation.json'
STEERING_COEFFICIENT = 0.229

def get_files(log_file):
    data = pd.read_csv(log_file)
    
    image_files = data['center'].tolist() + data['left'].tolist() + data['right'].tolist()
    
    for i, filename in enumerate(image_files):
        image_files[i] = os.path.basename(filename)
    
    targets = [np.array(data['steering']), 
               np.array(data['steering']) + STEERING_COEFFICIENT,
               np.array(data['steering']) - STEERING_COEFFICIENT]
    targets = np.concatenate(targets, axis=0).tolist()
    return image_files, targets


if __name__ == "__main__":
    # 1. Get image files & target labels    
    img_files, targets = get_files(LOG_FILE)
    
    # 2. Collect annotations
    annotations = []
    for filename, target in zip(img_files, targets):
        annotations.append({"filename": filename, "target": target})
    
    # 3. Write to annotation files
    with open('annotation.json', 'w') as fp:
        json.dump(annotations, fp, indent=4)


