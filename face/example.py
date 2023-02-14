import glob
import sys
sys.path.append('../')
from face.utils.dataset import ExamDataset
import torch
from torch.utils.data import DataLoader
from face.models.resnet import Resnet18
import torch.nn.functional as F

def init():
    path_list = glob.glob('./data/**/*.jpg', recursive=True)
    # path_list = glob.glob('/raid/lfeng/kz/test/photo/*.jpg', recursive=True)
    test_data_loader = DataLoader(ExamDataset(path_list), batch_size=1, shuffle=True, pin_memory=True)
    net = Resnet18()
    model_dict = net.state_dict()
    pretrained_dict = torch.load('./trained_weights/Downstream_train_om_test_ci.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()
    return net, test_data_loader

def predict():
    threshold = 0.0083
    net, test_data_loader = init()
    # t = tqdm(test_data_loader)
    net.eval()
    for b, (imgs, labels, path) in enumerate(test_data_loader):
        imgs = imgs.cuda()
        labels = labels.cuda().view(-1)
        out = net(imgs)
        prob = F.softmax(out, dim=1).cpu().data.numpy()[:, 1]
        if(prob>=threshold):
            print(path[0] + ':  Predict:1, lable:' + str(labels.cpu().data.numpy()))
        else:
            print(path[0] + ':  Predict:0, lable:' + str(labels.cpu().data.numpy()))

if __name__ == '__main__':
    predict()