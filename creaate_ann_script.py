# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os


LOG_FILES = ['C://Users//joonsup//git//dataset//logs//1.csv',
            'C://Users//joonsup//git//dataset//logs//2.csv',
            'C://Users//joonsup//git//dataset//logs//3.csv',
            'C://Users//joonsup//git//dataset//logs//4.csv',
            'C://Users//joonsup//git//dataset//logs//5.csv']
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
    acc_imgs = []
    acc_targets = []
    for log_file in LOG_FILES:
        img_files, targets = get_files(log_file)
        acc_imgs += img_files
        acc_targets += targets
    
    # 2. Collect annotations
    annotations = []
    for filename, target in zip(acc_imgs, acc_targets):
        annotations.append({"filename": filename, "target": target})
    
    # 3. Write to annotation files
    with open('annotation.json', 'w') as fp:
        json.dump(annotations, fp, indent=4)

    print(len(annotations))
