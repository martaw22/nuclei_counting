#This code will evaluate accuracy of the model

# evaluate the mask rcnn model on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
#from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from utils_myversion import compute_ap
 
# class that defines and loads the dataset
class NucleiDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "nuclei")
        # define data locations
        images_dir = dataset_dir + '/cropped_original_images/'
        annotations_dir = dataset_dir + '/xml_files/'
        # find all images
        for filename in listdir(images_dir):
            if filename == '.DS_Store':
                continue
            # extract image id
            image_id = filename[:-4]
            print('image_id', image_id)
            # skip bad images
            #if image_id in ['00090']:
            #    continue
            # skip all images after 150 if we are building the train set
            #if is_train and int(image_id) >= 150:
            #    continue
            # skip all images before 150 if we are building the test/val set
            #if not is_train and int(image_id) < 150:
            #    continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
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
 
# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "nuclei_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    box_errors = []
    #print('dataset imageid', dataset.image_id)
    for image_id in dataset.image_ids:
        print('image_id', image_id)
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        print('image size', image.size)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        print('rois', r['rois'])
        print('class ids', r['class_ids'])
        print('scores', r['scores'])
        print('gt bbox', gt_bbox)
        print('gt class id', gt_class_id)
        #calculate number of boxes as compared to predicted number of boxes
        #then calculate percent error
        num_boxes = len(r['rois'])
        pred_num_boxes = len(gt_bbox)
        if num_boxes == 0 and pred_num_boxes == 0:
            box_error = 0
        elif num_boxes == 0 and pred_num_boxes != 0:
            box_error = 1
        else:
            box_error = abs(pred_num_boxes-num_boxes)/num_boxes
        box_errors.append(box_error)
        print('num_boxes', num_boxes)
        print('pred_num_boxes', pred_num_boxes)
        print('box_error', box_error)

        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        print('AP', AP)
        if np.isnan(AP):
            AP = 0
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    mboxerrors = mean(box_errors)
    return mAP, mboxerrors
 
# load the train dataset
train_set = NucleiDataset()
train_set.load_dataset('nuclei', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
print('Train image_ids', train_set.image_ids)
# load the test dataset
test_set = NucleiDataset()
test_set.load_dataset('nuclei', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_nuclei_cfg_0005.h5', by_name=True)
# evaluate model on training dataset
#print(train_set.image_ids)

train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
#test_mAP = evaluate_model(test_set, model, cfg)
#print("Test mAP: %.3f" % test_mAP)

