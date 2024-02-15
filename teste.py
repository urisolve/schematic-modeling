import cv2
import os 
import torch
import torch as nn

import xml.etree.ElementTree as ET
from pathlib import Path

from torch_snippets import *
from torchvision import transforms
from sklearn.model_selection import train_test_split


current_path = os.getcwd()

new_path_images = r"dataset\drafter_1\images"
new_path_annotations = r"dataset\drafter_1\annotations"
new_path_segmentation = r"dataset\drafter_1\segmentation"

combined_path_image = os.path.join(current_path, new_path_images)
combined_path_annotations = os.path.join(current_path, new_path_annotations)
combined_path_segmentation= os.path.join(current_path, new_path_segmentation)
print(combined_path_image)
png_files = [file for file in os.listdir(combined_path_image) if file.endswith('.jpg') or file.endswith('.jpg') or file.endswith('.JPG')]
xml_files = [file for file in os.listdir(combined_path_annotations) if file.endswith('.xml')]

xml_file_path = os.path.join(combined_path_annotations, xml_files[0])
print(xml_files[0])
tree = ET.parse(xml_file_path)
root = tree.getroot()
# Suponha que você tem o nome do arquivo XML
xml_file_name = xml_files[0]

# Extraia o nome do arquivo sem a extensão
xml_name_without_extension = os.path.splitext(xml_file_name)[0]

# Construa o nome do arquivo PNG a partir do nome do arquivo XML
png_file_name = xml_name_without_extension + ".jpg"

print('nome da imagem',png_file_name)

        
image_path = os.path.join(combined_path_image, png_file_name)
image = cv2.imread(image_path)

# Iterar sobre cada objeto na anotação
for obj in root.findall('object'):
    name = obj.find('name').text
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    print("Objeto:", name)
    # print("Coordenadas da caixa delimitadora:", xmin, ymin, xmax, ymax)
        
#     # Desenhar caixa delimitadora na imagem
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
        
#     # Exibir o nome do objeto na caixa delimitadora
    cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,6, (0, 255, 0), 10)

h, w, c = image.shape
print (h,'  ', w, '  ',c )    

#     # Exibir a imagem com caixas delimitadoras

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Annotated Image', 1280, 815)

cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transforms():
  return transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]
                                 ) # for imagenet
                             ])
tfms = get_transforms()
print(image)
print (len(png_files))
print (len(xml_files))
