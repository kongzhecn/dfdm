import sys
sys.path.append('../')
import torch
from torch.utils.data import DataLoader
import glob
import torch.nn.functional as F
from fingerprint.utils.dataset import ExamDataset
from fingerprint.models.mobilenet import MobileNetV2

def init():
    path_list = glob.glob('./data/**/*.png', recursive=True)
    test_data = ExamDataset(path_list)
    # load the model parameters obtained from pretext task
    net = MobileNetV2()
    model_dict = net.state_dict()
    pretrained_dict = torch.load('./trained_weights/Downstream_train_Orcathus_test_ Orcathus.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()

    test_loader = DataLoader(test_data, batch_size=1 ,shuffle=True, pin_memory=True, num_workers=1)
    return net, test_loader

def predict():
    net, test_loader = init()
    net.eval()
    for b, (img, labels, path) in enumerate(test_loader):
        img = img.type(torch.FloatTensor).cuda()
        labels = labels.cuda().view(-1)
        out = net(img)
        out = F.softmax(out, dim=1)
        prob = torch.argmax(out, 1)
        print('%s:  Predict:%d, lable:%d' %(path[0], prob, labels.cpu().data.numpy()))

if __name__ == '__main__':
    predict()
