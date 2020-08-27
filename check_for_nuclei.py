'''This will check whether each cropped image has at least one nucleus in it.  If not, it will remove the unannotated image from the main folder so that images without any nuclei are not biasing the model.'''

import cv2
import numpy as np
import os
import shutil

class CheckNuclei:

    def __init__(self, path_annotated, path_unannotated, filename):
        '''Constructor'''
        self.path_annotated = path_annotated
        self.path_unannotated = path_unannotated
        self.filename = filename
        self.save_path = 'nuclei/val_data/cropped_images/no_nuclei/'


    def load_image(self):
        '''load the image'''
        image_annot = cv2.imread(self.path_annotated)
        image_unannot = cv2.imread(self.path_unannotated)
        return image_annot, image_unannot

    def check_nuclei(self, image_annot):
        '''Checks whether there is an annotation mark in the annotated image'''
        '''If there are nuclei present, returns True, else returns False'''

        #create boundaries for annotation color
        lower = np.array([200,0,200], dtype = 'uint8')
        upper = np.array([255,50,255], dtype = 'uint8')
        #find colors within the boundaries and apply the mask
        mask = cv2.inRange(image_annot, lower, upper)
        output = cv2.bitwise_and(image_annot, image_annot, mask=mask)
        if np.any(output):
            return True
        else:
            return False
        

    def move_file(self, image_unannot):
        '''Moves file from unannotated dir to no_nuclei dir'''
        shutil.move(self.path_unannotated, self.save_path + self.filename)


#load and check each image
dir_annotated = 'nuclei/val_data/cropped_images/annotated/'
dir_unannotated = 'nuclei/val_data/cropped_images/unannotated/'
for file in os.listdir(dir_annotated):
    print('file', file)
    if file == '.DS_Store':
        continue
    else:
        path_annotated = dir_annotated + file
        path_unannotated = dir_unannotated + file
        nuclei_file = CheckNuclei(path_annotated, path_unannotated, file)
        image_annot, image_unannot = nuclei_file.load_image()
        #nuclei_file.check_nuclei(image_annot)
        if nuclei_file.check_nuclei(image_annot) == False:
            print('no nuclei present')
            '''nuclei_file.move_file(image_unannot)'''
