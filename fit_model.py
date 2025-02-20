# fit a mask rcnn on the nuclei dataset

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import time
import random
import numpy as np
import pickle

# class that defines and loads the nuclei dataset
class NucleiDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
                self.add_class("dataset", 1, "nuclei")
		# define data locations
                images_dir = dataset_dir + '/cropped_images_pool/'
                annotations_dir = dataset_dir + '/xml_files/cropped/'
                #define validation data location
                #val_images_dir = dataset_dir + '/cropped_images_pool/'
                #val_annotations_dir = dataset_dir + '/xml_files/cropped/'
                #define random integer list
                images_list = listdir(images_dir)
                #select training images
                train_im = np.random.choice(images_list, int(len(images_list)*.9), replace = False)
                #get the remaining images for validation
                val_im = [i for i in images_list if i not in train_im]
                #save validation data so we can use it in eval
                with open('val_images_15epochs_0.01lr_2', 'wb') as im_list:
                        pickle.dump(val_im, im_list)
                # find all training images
                if is_train:
                        for filename in train_im:
                                if filename == '.DS_Store':
                                        continue
                                # extract image id
                                image_id = filename[:-4]
                                # skip bad images
                                #if image_id in ['00090']:
                                #	continue
                                # skip all images after 150 if we are building the train set
                                #if is_train and int(image_id) >= 150:
                                #	continue
                                # skip all images before 150 if we are building the test/val set
                                #if not is_train and int(image_id) < 150:
                                #	continue
                                img_path = images_dir + filename
                                ann_path = annotations_dir + image_id + '.xml'
                                # add to dataset
                                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
                #find all val images
                if not is_train:
                        for filename in val_im:
                                if filename == '.DS_Store':
                                        continue
                                #extract image id
                                image_id = filename[:-4]
                                img_path = images_dir + filename
                                ann_path = annotations_dir + image_id + '.xml'
                                #add to dataset
                                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
                        
	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('nuclei'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class NucleiConfig(Config):
        # define the name of the configuration
        NAME = "nuclei_cfg"
        # number of classes (background + nuclei)
        NUM_CLASSES = 1 + 1
        #Reduce batch size
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # number of training steps per epoch
        STEPS_PER_EPOCH = 2
        #images are cropped from originals
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        #size of fully connected layer
        #FPN_CLASSIF_FC_LAYERS_SIZE = 512
        # Max number of final detections                                                                                                   
        DETECTION_MAX_INSTANCES = 10000
        #learning rate and momentum - original were 0.001 and 0.9
        LEARNING_RATE = 0.01
        LEARNING_MOMENTUM = 0.9


#start timing the run
time_start = time.perf_counter()
print('time started', time_start)

# prepare train set
train_set = NucleiDataset()
train_set.load_dataset('nuclei', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = NucleiDataset()
test_set.load_dataset('nuclei', is_train=False)
test_set.prepare()
print('Validation: %d' % len(test_set.image_ids))
# prepare config
config = NucleiConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=15, layers='heads')
#save the model - can't use this, but saving checkpoint
#model.save('model_weights_020121')

#end time of run
time_end = time.perf_counter()

print(f"Model fitting took {time_end - time_start:0.4f} seconds")
