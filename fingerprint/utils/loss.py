import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, perceptual_net, train=False):
        super(PerceptualLoss, self).__init__()
        self.perceptual_net = perceptual_net
        self.criterion = nn.MSELoss().cuda()
        self.train = train
    def forward(self, x, y):
        y1, y2, y3, y4, y5 = self.perceptual_net.get_feature(y)
        x1, x2, x3, x4, x5, = self.perceptual_net.get_feature(x)
        loss = self.criterion(x1,y1)+self.criterion(x2, y2)+self.criterion(y3,y3)
        loss = self.criterion(x,y) + loss/3.
        return loss
    def get_net_parameters(self):
        return self.net.parameters()
    def get_perceptual_net_parameters(self):
        return self.perceptual_net.parameters()