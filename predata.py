import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from function import *

class circuitDataset(Dataset):
    def __init__(self, split, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir 
        self.mask_dir = mask_dir 
        self.transform = transform
        self.items = os.listdir(image_dir)
    
    ## return nº of samples in dataset
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.items[index])
        mask_path = os.path.join(self.mask_dir, self.items[index]) 
        
        image = cv2.imread(image_path)
        print()
        # print ("path da imagem", image_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
        image = cv2.resize(image, (224,224))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # print ("path da mascara", mask_path)
        
        mask = cv2.resize(mask, (224,224),interpolation=cv2.INTER_NEAREST) ##224
        print()
        # Calcular os valores únicos na imagem
        unique_values = len(set(mask.flatten()))
        # print()
        # print()
        # print()
        # print("A imagem tem", unique_values, "tons de cinza diferentes.")
        
        # print('olalalalaalalalalalal')
        # print('olalalalaalalalalalal')
        # print('olalalalaalalalalalal')
        # print('olalalalaalalalalalal')
        
        
        if self.transform  is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"] 
        return image, mask
        
    def choose(self):
        return self[randint(len(self))]
    

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        print('Valores das máscaras antes da normalização:')
        for mask in masks:
            print(torch.unique(torch.from_numpy(mask)))
        
        # Convertendo as imagens para tensores e normalizando
        ims = torch.stack([torch.tensor(im, dtype=torch.float32) / 255. for im in ims])
        
        # Mapeando os valores de pixel das máscaras para o intervalo de 0 a 12
        mapped_masks = []
        for mask in masks:
            mapped_mask = torch.zeros_like(torch.from_numpy(mask))
            for pixel_value, class_value in pixel_to_class.items():
                if isinstance(pixel_value, int):
                    mapped_mask[mask == pixel_value] = class_value
                elif isinstance(pixel_value, tuple):
                    mapped_mask[(mask >= pixel_value[0]) & (mask <= pixel_value[1])] = class_value
            mapped_masks.append(mapped_mask)
        mapped_masks = torch.stack(mapped_masks)
        
        print('Valores das máscaras após a normalização:')
        for mask in mapped_masks:
            print(torch.unique(mask))
        
        return ims, mapped_masks