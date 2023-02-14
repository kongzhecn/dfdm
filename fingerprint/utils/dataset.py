from torch.utils.data import Dataset
from torchvision import transforms
import skimage.io as io
import numpy as np

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])

class LivDetDataset(Dataset):
    def __init__(self,txtpath,search,transform=preprocess):
        with open(txtpath, mode='r') as ftxt:
            pathl = ftxt.readlines()
        imgs = []
        for row in pathl:
            row = row.replace('\n','')
            cond = True
            for s in search:
                if s not in row:
                    cond = False
            if cond:
                if 'Live' in row:
                    imgs.append([row,0])
                if 'Fake' in row:
                    imgs.append([row,1])
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fp, label = self.imgs[index]
        img = io.imread(fp).astype(np.float32)
        img = 1.0 * img / 255
        if self.transform is not None:
            if len(img.shape)==2:
                img = img.reshape(img.shape[0],img.shape[1],1)
            if img.shape[-1]==1:
                img = np.tile(img,(1,1,3))
            if img.shape[-1]==4:
                img = img[:,:,:3]
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class ExamDataset(Dataset):
    def __init__(self,list,transform=preprocess):
        self.list = list
        self.transform = transform

    def __getitem__(self, index):
        fp = self.list[index]
        if (fp.find('live') != -1):
            img_label = 0
        else:
            img_label = 1
        img = io.imread(fp).astype(np.float32)
        img = 1.0 * img / 255
        if self.transform is not None:
            if len(img.shape)==2:
                img = img.reshape(img.shape[0],img.shape[1],1)
            if img.shape[-1]==1:
                img = np.tile(img,(1,1,3))
            if img.shape[-1]==4:
                img = img[:,:,:3]
            img = self.transform(img)
        return img,img_label, fp

    def __len__(self):
        return len(self.list)