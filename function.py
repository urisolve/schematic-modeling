import torch
import torch.nn as nn
import cv2
import os
from torch_snippets import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.models import vgg16_bn


pixel_to_class = {
    0: 0,
    (99, 101): 1,
    (102, 108): 2,
    (109, 112): 3,
    (113, 137): 4,
    (138, 173): 5,
    (174, 182): 6,
    (183, 190): 7,
    (191, 200): 8,
    (201, 214): 9,
    215: 10
}

# Função para encontrar a classe correspondente para um valor de pixel
def find_class(pixel_value, pixel_to_class):
    for key, value in pixel_to_class.items():
        if isinstance(key, int):
            if pixel_value == key:
                return value
        elif isinstance(key, tuple):
            if key[0] <= pixel_value <= key[1]:
                return value
    return 11

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )
    
def UnetLoss(preds, targets):
    # Convertendo os rótulos para o tipo de dados correto (torch.long)
    targets = targets.type(torch.long)
    
    # Calculando a perda de entropia cruzada
    ce_loss = nn.CrossEntropyLoss()(preds, targets)
    
    # Calculando a precisão (accuracy)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    
    return ce_loss, acc

def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    
    # Transpor as dimensões para corresponder à ordem esperada pela rede neural
    ims = ims.permute(0, 3, 1, 2)
    
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()
@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    
    # Transpor as dimensões para corresponder à ordem esperada pela rede neural
    ims = ims.permute(0, 3, 1, 2)
    
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()

def get_transforms():
  return transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]
                                 ) # for imagenet
                             ])
tfms = get_transforms()