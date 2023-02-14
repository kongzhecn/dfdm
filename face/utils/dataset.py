from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class IfomDataset(Dataset):
    def __init__(self, data, transforms=None, train=True):
        self.data = data
        self.train = train
        if transforms is None:
            if self.train:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor()
                ])
            else:
                self.transforms = T.Compose([
                    T.ToTensor()
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]['photo_path']
        img_label = self.data[item]['photo_label']
        img = Image.open(img_path)
        videoID = self.data[item]['photo_belong_to_video_ID']
        img = self.transforms(img)
        return img, img_label, videoID

class ExamDataset(Dataset):
    def __init__(self, list, transforms=None):
        self.list = list
        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor()
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        img_path = self.list[item]
        if(img_path.find('real')!=-1):
            img_label = 1
        else:
            img_label = 0
        img = Image.open(img_path)
        img = self.transforms(img)
        return img, img_label, img_path