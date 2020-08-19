'''Script to preprocess the images before they are run by the model'''

import cv2
import numpy as np
import os

#first want to open an image and print out what size it is
image_path = 'nuclei/val_data/34.7_225_paa30x_17jan2020t1400_annotated.png'
image = cv2.imread(image_path)
height, width = image.shape[:2]
print('height', height)
print('width', width)
