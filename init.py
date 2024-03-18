import torch
import torch.nn as nn
import cv2
import os
from torch_snippets import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.models import vgg16_bn
from function import *
from model import *
from torch_snippets import subplots 
      
pathformodel = r'D:\Utilizadores\Diogo Moreira\Desktop\Tese\new_pc\code3\u-net'
# imagedi =r'D:\Utilizadores\Diogo Moreira\Desktop\dataset1cv\images_prepped_train'
# maskdi = r'D:\Utilizadores\Diogo Moreira\Desktop\dataset1cv\annotations_prepped_train'

imagedi =r'D:\Utilizadores\Diogo Moreira\Desktop\DATASET_1\task_segmentation1_annotations_2024_03_06_19_37_47_segmentation mask 1.1\train'
maskdi = r'D:\Utilizadores\Diogo Moreira\Desktop\DATASET_1\task_segmentation1_annotations_2024_03_06_19_37_47_segmentation mask 1.1\SegmentationClass\imagesimulator'
val_imagedir = r'D:\Utilizadores\Diogo Moreira\Desktop\validation\images'
val_maskdir = r'D:\Utilizadores\Diogo Moreira\Desktop\DATASET_2\task_validation_annotations_2024_03_13_17_18_13_segmentation mask 1.1\SegmentationClass'
trn_ds = circuitDataset('train',imagedi, maskdi)
val_ds = circuitDataset('test',val_imagedir, val_maskdir)
batch_size = 2


trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=val_ds.collate_fn)

print(trn_ds.__len__())



model = UNet().to(device)


criterion = UnetLoss

optimizer = optim.Adam(model.parameters(), lr=1e-3)
print ('optimizer.............', optimizer)
n_epochs =1

# Uncomment:
# model.encoder


# Train the model over increasing epochs
log_train_loss = []
log_train_acc = []
log_val_loss = []
log_val_acc = []

for ex in range(n_epochs):
    N = len(trn_dl)
    loss_avg = 0
    acc_avg = 0
    for bx, data in enumerate(trn_dl):
        # print('Batch:', bx)
        # print('Data shape:', data[0].shape)  # Imprime a forma do tensor de entrada
        # print('Data min value:', data[0].min().item())  # Imprime o valor mínimo do tensor de entrada
        # print('Data max value:', data[0].max().item())  # Imprime o valor máximo do tensor de entrada
        
        loss, acc = train_batch(model, data, optimizer, criterion)
        print(ex+(bx+1)/N,"  Loss=", loss, "Accuracy=", acc)
        loss_avg+=loss;
        acc_avg += batch_size * acc;
    log_train_loss.append(loss_avg/N)
    log_train_acc.append(acc_avg/(N*batch_size))

    N = len(val_dl)
    loss_avg = 0
    acc_avg = 0
    for bx, data in enumerate(val_dl):
        
        loss, acc = validate_batch(model, data, criterion)
        loss_avg+=loss;
        acc_avg += batch_size * acc;
        print(ex+(bx+1)/N,"  Loss=", loss, "Accuracy=", acc)
    log_val_loss.append(loss_avg/N)
    log_val_acc.append(acc_avg/(N*batch_size))
    
# torch.save(model.state_dict(), pathformodel)


plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.plot(log_train_loss, linestyle = 'dotted',  label='Loss train')
plt.plot(log_val_loss, linestyle = 'dotted',  label='Loss val')
plt.plot(log_train_acc, label='Acc train')
plt.plot(log_val_acc, label='Acc val')
plt.legend(loc='lower right')
plt.show()

# # Obtain a sample
# im, mask = next(iter(val_dl))

# # Feedforward
# _mask = model(im)

# _, _mask = torch.max(_mask, dim=1)
# subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
# ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],
# nc=3, titles=['Original image','Original mask','Predicted mask'])


im, mask = next(iter(val_dl))
print ('imagem',im.shape)
im = im.permute(0, 3, 1, 2)
print()
print('mascara',mask)
# Fazer a infer,ência com o modelo
_mask = model(im)
print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')
# Converter a saída da máscara prevista para a classe correspondente
_, _mask = torch.max(_mask, dim=1)
print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

print('chega aquiaiaiaiaiaiaiaiai')

# Verificar se o tensor de máscara tem apenas 2 dimensões
if len(mask[0].shape) == 2:
    # Se sim, adicionar uma dimensão para os canais
    mask[0] = mask[0].unsqueeze(0)

# Converter tensores em arrays numpy e reorganizar dimensões
im_cpu = im[0].permute(1, 2, 0).cpu().numpy()
mask_cpu = mask[0].cpu().numpy()
predicted_mask_cpu = _mask[0].cpu().numpy()

# Normalizar as imagens (se necessário)
im_cpu = cv2.normalize(im_cpu, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
mask_cpu = cv2.normalize(mask_cpu, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
predicted_mask_cpu = cv2.normalize(predicted_mask_cpu, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
predicted_mask_cpu =cv2.resize(predicted_mask_cpu, (512,512))
# Visualizar imagens usando cv2.imshow
cv2.imshow('Original image', im_cpu)
cv2.imshow('Original mask', mask_cpu)
cv2.imshow('Predicted mask', predicted_mask_cpu)
cv2.waitKey(0)
cv2.destroyAllWindows()