'''The complete example of loading the trained model and making a prediction for the first few images in the train and test datasets is listed below.'''

# detect nuclei in photos with mask rcnn model
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset

# class that defines and loads the kangaroo dataset
class NucleiDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "nuclei")
        # define data locations
        images_dir = dataset_dir + '/training_data/cropped_images/unannotated/'
        annotations_dir = dataset_dir + '/xml_files/cropped/'
        val_images_dir = dataset_dir + '/val_data/cropped_images/unannotated/'
        # find all images
        if is_train:
            for filename in listdir(images_dir):
                if filename == '.DS_Store':
                    continue
                # extract image id
                image_id = filename[:-4]
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
        if not is_train:
            for filename in listdir(val_images_dir):
                if filename == '.DS_Store':
                    continue
                #extract image id
                image_id = filename[:-4]
                #image path
                img_path = val_images_dir + filename
                ann_path = annotations_dir + image_id + '.xml'
                #add to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
                
    # load all bounding boxes for an image
    def extract_boxes(self, filename):
        # load and parse the file
        root = ElementTree.parse(filename)
        boxes = list()
        # extract each bounding box
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

    def return_id(self, image_id):
        return image_id
    
# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "nuclei_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=200):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        image_id = dataset.return_id(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        #pyplot.subplot(n_images, 2, i*2+1)
        #pyplot.figure()
        # plot raw pixel data
        #pyplot.imshow(image)
        #pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            print('j', j)
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        #pyplot.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='green')
            # draw the box
            ax.add_patch(rect)
        # show the figure
        save_path = 'image_preds/'
        pyplot.savefig(save_path + str(image_id) + '_bbox_preds_5epochs.png')

# load the train dataset
train_set = NucleiDataset()
train_set.load_dataset('nuclei', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the val dataset
test_set = NucleiDataset()
test_set.load_dataset('nuclei', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_folder = 'nuclei_cfg20210201_10epochs/'
model_path = model_folder + 'mask_rcnn_nuclei_cfg_0005.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
#plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)


