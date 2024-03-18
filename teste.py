import numpy as np
import torch
from PIL import Image
import cv2
import os

# def mask_to_tensor(mask, num_classes=255):
#     """
#     Convert a mask represented as an image to a tensor with specified number of classes.

#     Args:
#     - mask (numpy.ndarray): The input mask represented as an image.
#     - num_classes (int): The total number of classes.

#     Returns:
#     - torch.Tensor: The tensor representation of the mask.
#     """
#     # Convert the mask to a numpy array if it's not already
#     mask_np = np.array(mask)

#     # Create an empty tensor filled with zeros
#     tensor = torch.zeros(mask_np.shape, dtype=torch.long)

#     # Convert each unique value in the mask to a class index
#     unique_values = np.unique(mask_np)
#     for i, value in enumerate(unique_values):
#         # If the value is 0, it represents the background and should be skipped
#         if value == 0:
#             continue
#         # Assign the class index to all pixels with this value in the mask
#         tensor[mask_np == value] = i

#     return tensor


# mask_image = cv2.imread((r'D:\Utilizadores\Diogo Moreira\Desktop\DATASET_1\task_segmentation1_annotations_2024_03_06_19_37_47_segmentation mask 1.1\SegmentationClass\imagesimulator\1.png'))

# # Convert BGR image to RGB
# mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

# # Convert the mask image to a tensor
# tensor = mask_to_tensor(mask_image_rgb)

# # Convert tensor to numpy array
# tensor_np = tensor.numpy()

# print("Tensor:")
# print(tensor_np)

# max_value = np.max(tensor_np)

# # Encontre o valor mínimo
# min_value = np.min(tensor_np)

# print("Máximo valor:", max_value)
# print("Mínimo valor:", min_value)mask_dir = 'path_to_your_mask_directory'

# Initialize a variable to store the maximum label value
max_label = -1
mask_dir =r'D:\Utilizadores\Diogo Moreira\Desktop\DATASET_1\task_segmentation1_annotations_2024_03_06_19_37_47_segmentation mask 1.1\SegmentationClass\imagesimulator'

# Iterate through each mask in the directory
for mask_file in os.listdir(mask_dir):
    # Construct the full path to the mask file
    mask_path = os.path.join(mask_dir, mask_file)
    
    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find the maximum pixel value (which corresponds to the maximum label)
    current_max = mask.max()
    
    # Update max_label if the current_max is greater
    if current_max > max_label:
        max_label = current_max

# Print the maximum label value
print("Maximum label value in the dataset:", max_label)




# ims, masks = list(zip(*batch))
#         print ('entra aqui ')
#         print ('entra aqui ')
#         print ('entra aqui ')
#         print ('entra aqui ')
#         print ('entra aqui ')
#         print ('entra aqui ')
#         print ('entra aqui ')
#         print ('entra aqui ')
#         ims = torch.cat([tfms(im.copy()/255.)[None] for im in ims]).float().to(device)
#         ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
#         print()
#         print()
#         print('valores mascaras')
#         print(ce_masks)
#         return ims, ce_masks


def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        print('Valores das máscaras antes da concatenação:')
        for mask in masks:
            print(torch.unique(torch.from_numpy(mask)))
        
        ims = torch.cat([tfms(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        # print(ce_masks.values)
        return ims, ce_masks