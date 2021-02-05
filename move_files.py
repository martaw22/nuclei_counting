#move all the images from the annotated folder that do not have corresponding .xml files

import os
import shutil

#data paths for images
train_path = 'nuclei/training_data/cropped_images/annotated/'
val_path = 'nuclei/val_data/cropped_images/annotated/'
xml_path = 'nuclei/xml_files/cropped/'

#destination folder
dest_path = 'nuclei/annotated_no_xml/'

#get files
xml_files = os.listdir(xml_path)
train_files = os.listdir(train_path)
val_files = os.listdir(val_path)

#make list of all xml files
xml_names = []
for file in xml_files:
    xml_names.append(file[:-4])

#if train file name is not in the xml list, remove it
for file in train_files:
    name = file[:-4]
    if name not in xml_names:
        shutil.move(os.path.join(train_path, file), dest_path)
        
#same again for validation 
val_names = []
for file in val_files:
    name = file[:-4]
    if name not in xml_names:
        shutil.move(os.path.join(val_path, file), dest_path)

