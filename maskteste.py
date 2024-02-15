import os
import cv2
import numpy as np

import xml.etree.ElementTree as ET
from pathlib import Path


all_boxes = []

def generate_masks(image_path, bounding_boxes):
    # Read the image
    image = cv2.imread(image_path)
    
    # Initialize an empty mask
    mask = np.zeros_like(image[:, :, 0])
    
    # Loop through each bounding box
    for box in bounding_boxes:
        # Extract coordinates
        x_min, y_min, x_max, y_max = box
        
        # Draw filled rectangle on the mask
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255), -1)
    
    return mask



current_path = os.getcwd()

new_path_images = r"dataset\drafter_1\images"
new_path_annotations = r"dataset\drafter_1\annotations"
new_path_segmentation = r"dataset\drafter_1\segmentation"

combined_path_image = os.path.join(current_path, new_path_images)
combined_path_annotations = os.path.join(current_path, new_path_annotations)

xml_files = [file for file in os.listdir(combined_path_annotations) if file.endswith('.xml')]

xml_file_path = os.path.join(combined_path_annotations, xml_files[0])

tree = ET.parse(xml_file_path)
root = tree.getroot()
# Suponha que você tem o nome do arquivo XML
xml_file_name = xml_files[0]

# Extraia o nome do arquivo sem a extensão
xml_name_without_extension = os.path.splitext(xml_file_name)[0]

# Construa o nome do arquivo PNG a partir do nome do arquivo XML
png_file_name = xml_name_without_extension + ".jpg"

image_path = os.path.join(combined_path_image, png_file_name)
image = cv2.imread(image_path)

print('nome da imagem',png_file_name)
print('numero de ficheiros xml', len(xml_file_path))
print(xml_files[0])


for obj in root.findall('object'):
    name = obj.find('name').text
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    print(name, xmin,ymin, xmax, ymax)
    
    all_boxes.append((xmin, ymin, xmax, ymax))
    
all_boxes = np.array(all_boxes)



mask = generate_masks(image_path, all_boxes)
cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Annotated Image', 1280, 815)
print(mask)

cv2.imshow('Annotated Image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
