import torch
from torchvision import transforms
import torch.nn.functional as F
import random
from numpy import flipud, fliplr
import numpy as np
import copy


unit_proccess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
                 ])

rotation = transforms.Compose([
    transforms.Lambda(lambda images: torch.stack([unit_proccess(image) for image in images]))
                             ])

def image_Folding(x):
    x = x.detach().clone()
    n,c,h,w = x.size()
    center_x = torch.normal(mean=torch.tensor(0.5), std=torch.tensor(0.05))
    center_y = torch.normal(mean=torch.tensor(0.5), std=torch.tensor(0.05))

    center_x = torch.clamp(center_x,min=0.3,max=0.7)
    center_y = torch.clamp(center_y,min=0.3,max=0.7)
    center_x = int(h*center_x)
    center_y = int(w*center_y)

    patch1 = x[:,:,:center_x,:center_y]
    patch2 = x[:,:,center_x:,:center_y]
    patch3 = x[:,:,:center_x,center_y:]
    patch4 = x[:,:,center_x:,center_y:]

    patch_size = (patch1.size(2),patch1.size(3))

    patch1 = rotation(patch1)
    patch2 = F.interpolate(rotation(patch2),size=patch_size)
    patch3 = F.interpolate(rotation(patch3),size=patch_size)
    patch4 = F.interpolate(rotation(patch4),size=patch_size)

    patch = (patch1 + patch2 + patch3 + patch4) / 4.

    return F.interpolate(patch,size=(h,w))

def image_shuffling(x):
    n,c,h,w = x.size()
    x = x.detach().clone()
    x_perm = x[torch.randperm(n)]
    return x_perm

def cut_out(x, nholes, length, prob=0.5):
    if random.random() >= prob:
        return x
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    for i in range(img_batches):
        mask = np.ones((img_rows,img_cols))
        for n in range(nholes):
            c_x = np.random.randint(img_cols)
            c_y = np.random.randint(img_rows)

            y1 = np.clip(c_y - length // 2, 0, img_rows)
            y2 = np.clip(c_y + length // 2, 0, img_rows)
            x1 = np.clip(c_x - length // 2, 0, img_cols)
            x2 = np.clip(c_x + length // 2, 0, img_cols)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask.astype(np.float32))
        mask = mask.expand_as(imgs[i,:,:,:])
        imgs[i,:,:,:] = mask * imgs[i,:,:,:]
    return imgs

def data_augment(x,prob=0.5):
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    for i in range(img_batches):
        im = imgs[i,:,:,:].transpose(0, 2)
        im = im.numpy()
        if random.random() >= prob:
            im = fliplr(im)
        if random.random() >= prob:
            im = flipud(im)
        im = im.copy()
        im = torch.from_numpy(im)
        im = im.transpose(2, 0).unsqueeze(0)
        imgs[i,:,:,:] = im
    return imgs

