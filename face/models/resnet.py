import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        model_resnet = ResNet(BasicBlock, [2, 2, 2, 2])
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.dropout = nn.Dropout(0.5)


    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        self.dropout(feature)
        out = self.classifier_layer(feature)
        return out

    def get_feature(self, input):
        feature_maps = []
        feature = self.conv1(input) # 64, 128, 128
        feature = self.bn1(feature) # 64, 128, 128
        feature = self.relu(feature) # 64, 128, 128
        feature_maps.append(feature)
        feature = self.maxpool(feature) # 64, 64, 64
        feature = self.layer1(feature) # 64, 64, 64
        feature_maps.append(feature)
        feature = self.layer2(feature) # 128, 32, 32
        feature_maps.append(feature)
        feature = self.layer3(feature) # 256, 16, 16
        feature_maps.append(feature)
        feature = self.layer4(feature) # 512, 8, 8
        feature_maps.append(feature)
        feature = self.avgpool(feature) # 512, 1, 1
        feature = feature.view(feature.size(0), -1) # 10, 512
        self.dropout(feature)
        out = self.classifier_layer(feature)
        return feature_maps

    def get_code(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        return feature


class Discriminator_resnet18(nn.Module):
    def __init__(self):
        super(Discriminator_resnet18, self).__init__()
        model_resnet = ResNet(BasicBlock, [2, 2, 2, 2])
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(512, 1)
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0.0)
        self.dropout = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        self.dropout(feature)
        out = self.linear(feature)
        out = self.sig(out)
        return out