'''Script to preprocess the images before they are run by the model - resizes them and separately crops them into equal squares'''

import cv2
import numpy as np
import math
import os

class ImagePreprocessing:
    
    
    def __init__(self, path_annotated, path_unannotated, filename):
        '''Constructor'''
        self.path_annotated = path_annotated
        self.path_unannotated = path_unannotated
        self.filename = filename[:-4]
        self.save_path_resize = 'resized_images/'
        self.save_path_crop = 'cropped_images/'
        self.height_orig = None
        self.width_orig = None
        
    def load_image(self):
        '''load the image'''
        image_path = self.path_annotated
        image_annot = cv2.imread(image_path)
        image_unannot = cv2.imread(self.path_unannotated)
        self.height_orig, self.width_orig = image_annot.shape[:2]
        return image_annot, image_unannot

    def resize_image(self, image_annot, image_unannot):
        '''Resize the original image to 512x512 and add padding'''
        #choose percent by which to resize the image
        scale_percent = 20
        #calculate this percent of original dimensions
        new_w = int(self.width_orig * scale_percent / 100)
        new_h = int(self.height_orig * scale_percent / 100)
        #dsize
        dsize = (new_w, new_h)
        #resize image with same aspect ratio
        new_annot = cv2.resize(image_annot, dsize)
        new_unannot = cv2.resize(image_unannot, dsize)
        #now padd the resized image so that it is 512x512
        #this is hardcoded to apply to the standard image size in the nuclei dataset
        BLACK = [0,0,0]
        pad_annot = cv2.copyMakeBorder(new_annot,64,64,0,0,cv2.BORDER_CONSTANT,value=BLACK)
        pad_unannot = cv2.copyMakeBorder(new_unannot,64,64,0,0,cv2.BORDER_CONSTANT,value=BLACK)
        cv2.imwrite(self.save_path_resize + 'annotated/' + self.filename + '.png', pad_annot)
        cv2.imwrite(self.save_path_resize + 'unannotated/' + self.filename + '.png', pad_unannot)
        return pad_annot, pad_unannot

    def crop_image(self, image_annot, image_unannot):
        '''Crop the original image into equal sized 512x512 crops and add padding when necessary'''
        #choose size you want to end up with
        size = 512
        #find number of images that will be cropped out of height and width
        x = int(self.width_orig/size)
        y = math.ceil(self.height_orig/size)
        #set starting pixel value to crop at
        start_w = 0
        #loop through image, cropping first by width and then each of those crops by height
        for i in range(x):
            start_h = 0
            annot_crop_w = image_annot[:, start_w:(i+1)*size]
            unannot_crop_w = image_unannot[:, start_w:(i+1)*size]
            start_w += size
            for j in range(y):
                annot_crop_h = annot_crop_w[start_h:(j+1)*size, :]
                unannot_crop_h = unannot_crop_w[start_h:(j+1)*size, :]
                start_h += size
                #the last row of crops on the bottom of the image need padding on height
                if j == y-1:
                    BLACK = [0,0,0]
                    annot_crop_h = cv2.copyMakeBorder(annot_crop_h,64,64,0,0,cv2.BORDER_CONSTANT,value=BLACK)
                    unannot_crop_h = cv2.copyMakeBorder(unannot_crop_h,64,64,0,0,cv2.BORDER_CONSTANT,value=BLACK)
                cv2.imwrite(self.save_path_crop + 'annotated/' + self.filename + str(i) + str(j) + '.png', annot_crop_h)
                cv2.imwrite(self.save_path_crop + 'unannotated/' + self.filename + str(i) + str(j) + '.png', unannot_crop_h)
        return annot_crop_h, unannot_crop_h


#Load and process the images
dir_annotated = 'original_annotated_images/'
dir_unannotated = 'original_images/'
for file in os.listdir(dir_unannotated):
    print('file', file)
    if file == '.DS_Store':
        continue
    else:
        path_annotated = dir_annotated + file[:-4] + '_annotated.png'
        path_unannotated = dir_unannotated + file
        print('annotated file', path_annotated)
        print('original file', path_unannotated)

        nuclei_file = ImagePreprocessing(path_annotated, path_unannotated, file)
        image_annot, image_unannot = nuclei_file.load_image()
        pad_annot, pad_unannot = nuclei_file.resize_image(image_annot, image_unannot)
        crop_annot, crop_unannot = nuclei_file.crop_image(image_annot, image_unannot)


