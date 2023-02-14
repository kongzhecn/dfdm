import sys
sys.path.append('../')
import torch
from fingerprint.utils.dataset import LivDetDataset
import argparse
from torch.utils.data import DataLoader
from tqdm import *
import torch.nn.functional as F
import copy
import os
from torchvision.utils import save_image
from fingerprint.models.unet import Unet
from fingerprint.utils.loss import PerceptualLoss
from fingerprint.models.mobilenet import MobileNetV2, Discriminator_v2
from fingerprint.utils.utils import image_Folding, image_shuffling
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

switch = {
        'O': 'Orcathus',
        'G': 'GreenBit',
        'D': 'DigitalPersona'
    }
# parameters of pretext task
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root", type=str, default='./')
parser.add_argument("--train_sensor", type=str, default='D', help='Training dataset(G\D\O)')
parser.add_argument("--test_sensor", type=str, default='D', help='Testing dataset(G\D\O)')
parser.add_argument("--savedir", type=str, help='Save path')
parser.add_argument("--epoch", type=int, default=50, help='Training epoch number')
parser.add_argument("--lr", type=float, default=1e-6, help='learning rate')
parser.add_argument("--model", type=str, default='imagenet', help='Whether use pre-trained model')
parser.add_argument("--dataLabel_path", type=str, default='./data_path.txt', help='data label path')
parser.add_argument("--batchsize", type=int, default=12, help='Training batch size')
args = parser.parse_args()

t_train = switch[args.train_sensor]
t_test = switch[args.test_sensor]

# training initialization
def init():
    # load data
    try:
        train_data = LivDetDataset(args.dataLabel_path, [t_train, 'train'])
        test_data = LivDetDataset(args.dataLabel_path, [t_train, 'train'])
    except FileNotFoundError:
        print("Error: Please download the dataset.")
        sys.exit()
    if args.train_sensor == 'O':
        train_batch_size = 1
    else:
        train_batch_size = args.batchsize
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, pin_memory=True)

    # create network and load model of network
    model = MobileNetV2()
    if args.model == 'imagenet':
        try:
            model_dict = model.state_dict()
            pretrained_dict = torch.load('./pretrained_model/mobilenet_v2.pth.tar')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except FileNotFoundError:
            print("Error: The pretrained model will be made publicly available soon.")
            sys.exit()
    cls = Discriminator_v2()
    cls = cls.cuda()
    decoder = Unet(model)
    decoder = decoder.cuda()

    optimizer_IF = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_OM = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_D = torch.optim.Adam(cls.parameters(), lr=1e-3, weight_decay=5e-4)
    g_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_IF, 20, gamma=0.9, last_epoch=-1)

    return model, cls, decoder, train_loader, test_loader, optimizer_IF, optimizer_OM, optimizer_D, g_scheduler


def train():
    model, cls, decoder, train_loader, test_loader, optimizer_IF, optimizer_OM, optimizer_D, g_scheduler = init()

    acc_list = []

    for i in range(5):
        acc_list.append(0.05)


    loss_tr = 0
    loss_te = 0
    loss_val = 0
    acc_te = 0
    acc_val = 0
    tdr_val = 0
    tdr_te = 0
    model_ = copy.deepcopy(model)
    p_loss = PerceptualLoss(model_.eval())
    m_loss = nn.MSELoss(size_average=True)
    OM_loss = nn.MSELoss()


    for e in range(args.epoch):
        t = tqdm(train_loader)
        t.set_description("Whole Epoch [{}/{}]".format(e + 1, args.epoch))
        if args.train_sensor == 'O':
            init_cond = True
            train_cond = False
        opt_counter = 0
        test_counter = 0
        for b, (imgs, ls) in enumerate(t):
            decoder.train()
            model.train()
            # the size of images in Orcathus is different, resize the images to make the images in the same batch have the same size
            if args.train_sensor == 'O':
                if init_cond:
                    init_cond = False
                    train_cond = False
                    img_batches, img_deps, img_rows, img_cols = imgs.shape
                    if img_rows % 2 == 1:
                        img_rows = img_rows + 1
                    if img_cols % 2 == 1:
                        img_cols = img_cols + 1
                    imgs_list = []
                    l_list = []
                    counter = 0

                if counter == args.batchsize:
                    counter = 0
                    init_cond = True
                    train_cond = True
                else:
                    imgs = F.interpolate(imgs, size=[img_rows, img_cols], mode='bilinear', align_corners=False)

                    for im in imgs:
                        imgs_list.append(im.numpy())
                        l_list.append(ls.numpy())
                    counter += 1
                if train_cond:
                    imgs_list = torch.tensor(imgs_list)
                    l_list = torch.tensor(l_list)

                    imgs = imgs_list.type(torch.FloatTensor)
                    l = l_list.view(-1)

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
                    if (b % (args.batchsize * 3) == 0) or b == args.batchsize :
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
                    mix_z = alpha * z1 + (1 - alpha) * z2
                    mix_z = mix_z + torch.normal(mean=0., std=torch.ones_like(mix_z) * 0.1).cuda()
                    mixed_x = alpha * imgs + (1 - alpha) * shuffled_imgs
                    om_loss = OM_loss(mix_z, model.get_code(mixed_x).detach())
                    om_loss.backward()
                    optimizer_OM.step()
                    t.set_postfix_str('loss_IF: {:4f}, loss_OM: {:4f}, loss_g: {:4f}, loss_d: {:4f}'.format(
                        if_loss.item(), om_loss.item(), g_loss.item(), d_loss.item()
                    ))
            else:
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
                mix_z = alpha * z1 + (1 - alpha) * z2
                mix_z = mix_z + torch.normal(mean=0., std=torch.ones_like(mix_z) * 0.1).cuda()
                mixed_x = alpha * imgs + (1 - alpha) * shuffled_imgs
                om_loss = OM_loss(mix_z, model.get_code(mixed_x).detach())
                om_loss.backward()
                optimizer_OM.step()
                t.set_postfix_str('loss_IF: {:4f}, loss_OM: {:4f}, loss_g: {:4f}, loss_d: {:4f}'.format(
                    if_loss.item(), om_loss.item(), g_loss.item(), d_loss.item()
                ))
                t.update()
        if e % 1 == 0:
            # save model
            if not os.path.exists(args.root + 'pretext_model/' + args.savedir):
                os.makedirs(args.root + 'pretext_model/' + args.savedir)
            path = args.root + 'pretext_model/' + args.savedir + '/Train_' + t_train + '_test_' + t_test + '_epoch_' + str(e) + '.pth'
            torch.save(model.state_dict(), path)


if __name__ == "__main__":
    train()