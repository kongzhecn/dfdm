import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self,perceptual_net, train=False):
        super(PerceptualLoss, self).__init__()
        self.perceptual_net = perceptual_net
        self.criterion = nn.MSELoss().cuda()
        self.train = train
    def forward(self, x, y):
        la1,la2,la3,la4,la5 = self.perceptual_net.get_feature(y)
        out1,out2,out3,out4,out5, = self.perceptual_net.get_feature(x)
        loss = self.criterion(out1,la1)+self.criterion(out2,la2)+self.criterion(out3,la3)
        return loss/3.
    def get_perceptual_net_parameters(self):
        return self.perceptual_net.parameters()

