import sys
sys.path.append('../')
import argparse
import copy
import torch.nn as nn
from face.utils.utils import get_dataset, image_Folding, image_shuffling
from face.models.resnet import Resnet18, Discriminator_resnet18
from face.models.unet import Unet
from utils.loss import PerceptualLoss
import torch
import warnings
import os
from tqdm import *
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root", type=str, default='./')
parser.add_argument("--train_data", type=str, default='om', help='Trainint data(om/ci)')
parser.add_argument("--test_data", type=str, default='ci', help='Testing data(om/ci)')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size number')
parser.add_argument("--Epoch_num", type=int, default=50, help='Training epoch number')
parser.add_argument("--savedir",type=str, help='Save path')
parser.add_argument("--lr",type=float, default=1e-4, help='Learning rate')
parser.add_argument("--model", type=str, default='imagenet', help='Whether use pre-trained model')
args = parser.parse_args()

# training initialization
def init():
    # load data
    try:
        train_data_loader, test_data_loader = get_dataset('./labels', args.train_data, 1, args.test_data, 1, args.batch_size)
    except FileNotFoundError:
        print("Error: Please download the dataset.")
        sys.exit()
    model = Resnet18()

    # create network and load model of network
    if args.model == 'imagenet':
        try:
            model_dict = model.state_dict()
            pretrained_dict = torch.load('./pretrained_model/resnet18-5c106cde.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except FileNotFoundError:
            print("Error: The pretrained model will be made publicly available soon.")
            sys.exit()
    cls = Discriminator_resnet18().cuda()
    decoder = Unet(model).cuda()

    optimizer_IF = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_OM = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_D = torch.optim.Adam(cls.parameters(), lr=1e-3, weight_decay=5e-4)
    g_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_IF, 20, gamma=0.9, last_epoch=-1)
    return model, cls, decoder, train_data_loader, test_data_loader, optimizer_IF, optimizer_OM, optimizer_D, g_scheduler

def train():
    model, cls, decoder, train_data_loader, test_data_loader, optimizer_IF, optimizer_OM, optimizer_D, g_scheduler = init()
    model_ = copy.deepcopy(model)
    p_loss = PerceptualLoss(model_.eval())
    m_loss = nn.MSELoss(size_average=True)
    OM_loss = nn.MSELoss()

    for e in range(args.Epoch_num):
        decoder.train()
        model.train()
        t = tqdm(train_data_loader)
        t.set_description("Epoch [{}/{}]".format(e +1 ,args.Epoch_num))
        for b, (imgs, labels, _) in enumerate(t):
            # get the folded images
            folded_imgs = image_Folding(imgs)
            imgs = imgs.cuda()
            folded_imgs = folded_imgs.cuda()
            # Discriminator part
            optimizer_D.zero_grad()
            recon_imgs = decoder(folded_imgs)
            d_loss = -torch.mean(cls(imgs)) + torch.mean(cls(recon_imgs.detach()))
            d_loss.backward()
            optimizer_D.step()

            for p in cls.parameters():
                p.data.clamp_(-0.1, 0.1)

            optimizer_IF.zero_grad()
            recon_imgs = decoder(folded_imgs)
            # gan  part
            if b % 3 == 0:
                g_loss = -torch.mean(cls(recon_imgs))
                g_loss.backward(retain_graph=True)
            # if_loss part
            if e >= 1:
                if_loss = p_loss(recon_imgs, imgs) + m_loss(recon_imgs, imgs)
            else:
                if_loss = m_loss(recon_imgs, imgs)

            if_loss.backward()
            optimizer_IF.step()

            for p in decoder.parameters():
                p.data.clamp_(-0.8, 0.8)
            # operational consistency part
            shuffled_imgs = image_shuffling(imgs)
            optimizer_OM.zero_grad()

            z1 = model.get_code(imgs)
            z2 = model.get_code(shuffled_imgs)

            alpha = torch.randn(imgs.size(0), 1, 1, 1).cuda()
            mix_z = alpha * z1 + (1-alpha) * z2
            mix_z = mix_z + torch.normal(mean=0.,std=torch.ones_like(mix_z)*0.1)
            mixed_imgs = alpha * imgs + (1 - alpha) * shuffled_imgs
            om_loss = OM_loss(mix_z, model.get_code(mixed_imgs).detach())
            om_loss.backward()
            optimizer_OM.step()
            t.set_postfix_str('loss_IF: {:.4f}, loss_OM: {:.4f}, loss_g: {:.4f}, loss_d: {:.4f}'.format(
                if_loss.item(), om_loss.item(), g_loss.item(), d_loss.item()
            ))

        # save model
        if not os.path.exists(args.root + 'pretext_model/' + args.savedir):
            os.makedirs(args.root + 'pretext_model/' + args.savedir)
        save_path = args.root + 'pretext_model/' + args.savedir + '/Train_' + args.train_data + '_test_' + args.test_data + '_'+ str(e) + '.pth'
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train()

