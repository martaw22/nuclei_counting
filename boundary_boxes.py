'''Script to get a random crop from each image in a folder, save the crop of the original and annotated image, xml file of bboxes, and an image with bboxes.  The box will be of an arbitrary predetermined size.  The location will be based on pre-labeling of each nucleus with a dot of a contrasting color.'''

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

class NucleiAnnotation:

    def __init__(self, image_path_annotated, image_folder_not_annotated, image_filename):
        """
        Constructor
        """
        self.image_path_annotated = image_path_annotated
        self.image_path_not_annotated = image_folder_not_annotated
        self.filename = file[:-4]
        self.height_orig = None
        self.width_orig = None
        self.bboxes = None
        self.top_left_corner_h = None
        self.top_left_corner_w = None

    def load_image(self):
        #load the image
        #print('image path', image_path)
        image_orig = cv2.imread(self.image_path_annotated)
        #print('image orig', image_orig)
        image_no_annotations = cv2.imread(self.image_path_not_annotated)
        self.height_orig, self.width_orig = image_orig.shape[:2]
        #print('height', self.height_orig)
        #print('width', self.width_orig)
        return image_orig, image_no_annotations

    def crop_image(self, image_orig, image_no_annotations):
        #crop the image to be 512 by 512
        self.top_left_corner_h = np.random.randint(0, self.height_orig - 512)
        self.top_left_corner_w = np.random.randint(0, self.width_orig - 512)
        cropped_image = image_orig[self.top_left_corner_h:self.top_left_corner_h+512, self.top_left_corner_w:self.top_left_corner_w+512]
        cropped_no_annotations = image_no_annotations[self.top_left_corner_h:self.top_left_corner_h+512, self.top_left_corner_w:self.top_left_corner_w+512]
        cv2.imwrite('nuclei/cropped_original_images/' + self.filename + '_' + str(self.top_left_corner_h) + '_' + str(self.top_left_corner_w) + '_cropped.png', cropped_no_annotations)
        cv2.imwrite('cropped_annotated_images/' + self.filename + '_' + str(self.top_left_corner_h) + '_' + str(self.top_left_corner_w) + '_cropped.png', cropped_image)
        return cropped_image, cropped_no_annotations


    def create_color_boundaries(self):
        #create boundaries for the color of interest of the annotation color
        lower = np.array([200, 0, 200], dtype = 'uint8')
        upper = np.array([255, 50, 255], dtype = 'uint8')
        
        return lower, upper

    def find_contours(self, image_orig, lower, upper):
        #find the colors within the boundaries and apply the mask
        mask = cv2.inRange(image_orig, lower, upper)
        output = cv2.bitwise_and(image_orig, image_orig, mask=mask)

        #convert to grayscale and threshold so that everything gray becomes white
        imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)

        #find contours
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return thresh, contours
    
    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


    def create_xml_file(self):
        # create the file structure
        annotation = ET.Element('annotation')
        folder = ET.SubElement(annotation, 'folder').text = 'nuclei'
        filename = ET.SubElement(annotation, 'filename').text = self.image_path_annotated
        path = ET.SubElement(annotation, 'path').text = self.image_path_annotated
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(512)
        ET.SubElement(size, 'height').text = str(512)
        ET.SubElement(size, 'depth').text = str(3)
        segmented = ET.SubElement(annotation, 'segmented').text = '0'
        return annotation

    def draw_contours(self, thresh, contours, image_no_annotations, annotation):
        
        #draw contours
        # Approximate contours to polygons + get bounding rects and circles
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        #radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
           
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
        self.bboxes = image_no_annotations
        print('image no annotations', image_no_annotations.shape)
        print('bboxes size', self.bboxes.shape)
        # Draw polygonal contour + bonding rects
        for i in range(len(contours)):
            color = (0, 0, 255)
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0])-10, int(boundRect[i][1])+12),(int(boundRect[i][0]+boundRect[i][2]) + 10, int(boundRect[i][1]+boundRect[i][3]) - 12), color, 2)
            xmin = int(boundRect[i][0])-5
            ymax = int(boundRect[i][1])+7
            xmax = int(boundRect[i][0]+boundRect[i][2])+5
            ymin = int(boundRect[i][1]+boundRect[i][3])-7
            cv2.rectangle(self.bboxes, (xmin, ymax),(xmax, ymin), color, 2)
            #Add boundary box info to the xml file
            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = 'nucleus'
            bx = ET.SubElement(object, 'bndbox')
            ET.SubElement(bx, 'xmin').text = str(xmin)
            ET.SubElement(bx, 'ymin').text = str(ymin)
            ET.SubElement(bx, 'xmax').text = str(xmax)
            ET.SubElement(bx, 'ymax').text = str(ymax)

        # create a new XML file with the results
        mydata = self.prettify(annotation)
        filename = self.image_path_not_annotated[:-4]
        #myfile = open('nuclei/xml_files/' + self.filename + '_' + str(self.top_left_corner_h) + '_' + str(self.top_left_corner_w) + '_cropped.xml', "w")
        myfile = open('nuclei/xml_files/cropped/' + self.filename + '.xml', 'w')
        myfile.write(mydata)
        return drawing

    def save_bbox_image(self, drawing):
        #cv2.imwrite ('test_mask.png', output)
        #cv2.imwrite('test_contours.png', im2)
        #cv2.imshow('grayscale', imgray)
        #cv2.imwrite('bboxdrawing.png', drawing)
        #cv2.imshow('bbox', self.bboxes)
        #cv2.imwrite('nuclei/training_data/resized_images/bbox_images/' + self.filename + '_' + str(self.top_left_corner_h) + '_' + str(self.top_left_corner_w).png', self.bboxes)
        cv2.imwrite('nuclei/training_data/cropped_images/bbox_images/' + self.filename + '.png', self.bboxes)

path = 'nuclei/training_data/cropped_images/unannotated/'
otherdirectory = 'nuclei/training_data/cropped_images/annotated/'
for file in os.listdir(path):
    print('file', file)
    if file == '.DS_Store':
        continue
    else:
        annotated_filepath = otherdirectory + file[:-4] + '.png'
        original_filepath = path + file
        print('annoted file', annotated_filepath)
        print('original file', original_filepath)

        nuclei_file = NucleiAnnotation(annotated_filepath, original_filepath, file)
        image_orig, image_no_annotations = nuclei_file.load_image()
        #cropped_image, cropped_no_annotations = nuclei_file.crop_image(image_orig, image_no_annotations)
        lower, upper = nuclei_file.create_color_boundaries()
        thresh, contours = nuclei_file.find_contours(image_orig, lower, upper)
        annotation = nuclei_file.create_xml_file()
        drawing = nuclei_file.draw_contours(thresh, contours, image_no_annotations, annotation)
        nuclei_file.save_bbox_image(drawing)

'''
#show the images

cv2.imshow("images", np.hstack([image_orig, output]))
cv2.imshow('grayscale', imgray)
cv2.imshow('thresh', thresh)
cv2.imshow('image_contours', im2)
cv2.imshow('bbox', drawing)
cv2.imshow('bbox_orig', bboxes)
cv2.waitKey(0)
'''
