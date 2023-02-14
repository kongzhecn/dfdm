import sys
sys.path.append('../')
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import *
import torch.nn.functional as F
import os
import logging
from fingerprint.utils.dataset import LivDetDataset
from fingerprint.models.mobilenet import MobileNetV2
from fingerprint.utils.utils import cut_out, data_augment
from fingerprint.utils.evaluate import ACE_TDR_Cal
import numpy as np
from sklearn.metrics import roc_auc_score
from fingerprint.utils.evaluate import get_EER_states

parser = argparse.ArgumentParser(description='Manual to this script')
parser.add_argument("--root", type=str, default='./', help='Root directory')
parser.add_argument("--train_sensor", type=str, default='O', help='Training set(G\D\O)')
parser.add_argument("--test_sensor", type=str, default='O', help='Testing set(G\D\O)')
parser.add_argument("--dataLabel_path", type=str, default='./data_path.txt', help='Path of data label')
parser.add_argument("--opt", type=str, default='Adam', help='Optimizer (SGD\Adam)')
parser.add_argument("--savedir",type=str, help='Save path')
parser.add_argument("--loadModelPath",type=str, help='The path of pretext task model')
parser.add_argument("--Epoch_num", type=int, default=90, help='Training epoch number')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate')
parser.add_argument("--criteria",type=str, default='TDR', help='Reference index for saving model')
parser.add_argument("--eval_epoch", type=int, default=2, help='Test frequency')
parser.add_argument("--batch_size", type=int, default=64, help='Batch size number')
parser.add_argument("--opt_num", type=int, default=4, help='Number of batches for gradient update')
parser.add_argument("--batch_size_for_test", type=int, default=2)
parser.add_argument("--test_batch_num", type=int, default=256)
parser.add_argument("--n_holes", type=int, default=10, help='Number of holes for cut out')
parser.add_argument("--length", type=int, default=96, help='Size of holes')
args = parser.parse_args()

switch = {
        'O': 'Orcathus',
        'G': 'GreenBit',
        'D': 'DigitalPersona'
    }
t_train = switch[args.train_sensor]
t_test = switch[args.test_sensor]


# log configuration
logger = logging.getLogger("mainModule")
logger.setLevel(level=logging.DEBUG)
if not os.path.exists(args.root + 'downstream_log/' + args.savedir):
    os.makedirs(args.root + 'downstream_log/' + args.savedir)
log_path = args.root + 'downstream_log/' + args.savedir + '/Train_' + t_train + '_test_' + t_test + '.txt'
handler = logging.FileHandler(filename=log_path, mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)


# save training parameters to log
logger.info("Training Protocol")
logger.info("log path:" + log_path)
logger.info("Epoch Total number:{}".format(args.Epoch_num))
logger.info("Training sensor: " + t_train)
logger.info("Testing sensor: " + t_test)
logger.info("Data label path: " + args.dataLabel_path)
logger.info("Train Batch Size is {:^.2f} x {:^.2f}".format(args.batch_size, args.opt_num))
logger.info("Test Batch Size is {:^.2f} x {:^.2f}".format(args.test_batch_num, args.batch_size_for_test))
logger.info("Eval epoch: {:^.2f}".format(args.eval_epoch))
logger.info("Optimizer is {}".format(args.opt))
logger.info("Learning Rate is {:^.4f}".format(args.learning_rate))
logger.info("load path:" + args.loadModelPath)
logger.info("Save path: " + args.savedir)
logger.info("Criteria:" + args.criteria)
logger.info("Cut out size if {:^.2f} x {:^.2f} x {:^.2f}".format(args.n_holes, args.length, args.length))

# training initialization
def init():
    try:
        train_data = LivDetDataset(args.dataLabel_path, [t_train ,'train'])
        val_data = LivDetDataset(args.dataLabel_path, [t_test ,'test'])
        test_data = LivDetDataset(args.dataLabel_path, [t_test ,'test'])
    except FileNotFoundError:
        print("Error: Please download the dataset.")
        sys.exit()

    # load the model parameters obtained from pretext task
    net = MobileNetV2()
    model_dict = net.state_dict()
    pretrained_dict = torch.load(args.loadModelPath)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()

    if args.train_sensor=='O':
        train_batch_size = 1
    else:
        train_batch_size = args.batch_size

    train_loader = DataLoader(train_data, batch_size=train_batch_size ,shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1 ,shuffle=True, pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1 ,shuffle=True, pin_memory=True, num_workers=1)
    return net, train_loader, val_loader, test_loader




def train():
    net, train_loader, val_loader, test_loader = init()
    net.cuda()
    net_loss = nn.CrossEntropyLoss().cuda()
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    crite_list = []
    for i in range(5):
        crite_list.append(0.05)
    loss_tr = 0
    loss_te = 0
    loss_val = 0
    acc_te = 0
    acc_val = 0
    tdr_val = 0
    tdr_te = 0
    for e in range(args.Epoch_num):
        t = tqdm(train_loader)
        t.set_description("Whole Epoch [{}/{}]".format(e +1 ,args.Epoch_num))
        if args.train_sensor == 'O':
            init_cond = True
            train_cond = False
        opt_counter = 0
        test_counter = 0
        test_dataiter = iter(test_loader)
        val_dataiter = iter(val_loader)
        for b ,(imgs, ls) in enumerate(t):
            net.train()
            if args.train_sensor == 'O':
                if init_cond:
                    init_cond = False
                    train_cond = False
                    img_batches, img_deps, img_rows, img_cols = imgs.shape
                    imgs_list = []
                    l_list = []
                    counter = 0

                if counter == args.batch_size:
                    counter = 0
                    init_cond =True
                    train_cond = True
                else:
                    imgs = F.interpolate(imgs, size=[img_rows ,img_cols], mode='bilinear', align_corners=False)
                    imgs = data_augment(imgs)
                    imgs = cut_out(imgs, args.n_holes, args.length)
                    for im in imgs:
                        imgs_list.append(im.numpy())
                        l_list.append(ls.numpy())
                    counter += 1
                if train_cond:
                    imgs_list = torch.tensor(imgs_list)
                    l_list = torch.tensor(l_list)
                    imgs = imgs_list.type(torch.FloatTensor).cuda()
                    l = l_list.cuda().view(-1)
                    out = net(imgs)
                    crite = net_loss(out ,l)
                    loss = crite.cpu().detach().data.numpy()
                    loss_tr = 0.6 * loss_tr + 0.4 * loss
                    crite.backward()
                    opt_counter += 1
            else:
                imgs = data_augment(imgs)
                imgs = cut_out(imgs, args.n_holes, args.length)
                imgs = imgs.type(torch.FloatTensor).cuda()
                l = ls.cuda().view(-1)
                out = net(imgs)
                crite = net_loss(out ,l)
                loss = crite.cpu().detach().data.numpy()
                loss_tr = 0.6 * loss_tr + 0.4 * loss
                crite.backward()
                opt_counter += 1
            if opt_counter == args.opt_num:
                optimizer.step()
                optimizer.zero_grad()
                opt_counter = 0
                test_counter += 1
            # due to the large amount of data in the test set, only a part of the images are selected for testing during training
            if test_counter == args.eval_epoch:
                net.eval()
                with torch.no_grad():
                    for ind_ in range(2):
                        correct = torch.zeros(1).squeeze().cuda()
                        total = torch.zeros(1).squeeze().cuda()
                        loss = torch.zeros(1).squeeze().cuda()
                        result = []
                        prob_list = []
                        label_list = []
                        for j in range(args.test_batch_num):
                            for b_i in range(args.batch_size_for_test):
                                if ind_ == 0:
                                    try:
                                        img, l = test_dataiter.next()
                                    except StopIteration:
                                        test_dataiter = iter(test_loader)
                                        img, l = test_dataiter.next()
                                else:  # ind=1
                                    try:
                                        img, l = val_dataiter.next()
                                    except StopIteration:
                                        val_dataiter = iter(val_loader)
                                        img, l = val_dataiter.next()
                                img = img.type(torch.FloatTensor).cuda()
                                l = l.cuda().view(-1)
                                out = net(img)
                                out_f = F.softmax(out, dim=1)
                                loss += net_loss(out, l)
                                pred = torch.argmax(out_f, 1)
                                correct += (pred==l).sum().float()
                                total += len(l)
                                result.append([l.cpu().detach().data.numpy()[0], out_f.cpu().detach().data.numpy()[0 ,1]])
                                prob_list = np.append(prob_list, out.cpu().detach().data.numpy()[0, 1])
                                label_list = np.append(label_list, l.cpu().detach().data.numpy()[0])
                        if ind_ == 0:
                            acc_te = (correct/total).cpu().detach().data.numpy()
                            loss_te = (loss/total).cpu().detach().data.numpy()
                            ace_te, tdr_te = ACE_TDR_Cal(result)
                            eer_te, threshold_te, _, _ = get_EER_states(prob_list, label_list)
                        else:
                            acc_val = (correct/total).cpu().detach().data.numpy()
                            loss_val = (loss/total).cpu().detach().data.numpy()
                            ace_val, tdr_val = ACE_TDR_Cal(result)
                            eer_val, threshold_te, _, _ = get_EER_states(prob_list, label_list)
                test_counter = 0
                logger.info('Ace:%5.2f  Tdr:%5.2f  Eer:%5.2f  Acc:%5.2f  crite_list:[%s]' %(ace_te*100, tdr_te*100, eer_te*100, acc_te*100, ''.join(str('{:5.2f} '.format(i*100)) for i in crite_list)))
                if args.criteria == 'TDR':
                    crit_save = tdr_te
                if args.criteria == 'ACC':
                    crit_save = acc_te
                if args.criteria == 'ACE':
                    crit_save = 1-ace_te
                if args.criteria == 'EER':
                    crit_save = 1-eer_te
                # save 5 good model and testing their performance on the whole testing set after training
                if crit_save >= min(crite_list):
                    crite_list[crite_list.index(min(crite_list))] = crit_save
                    crite_list = sorted(crite_list, reverse=True)
                    if not os.path.exists(args.root + 'downstream_model/' + args.savedir):
                        os.makedirs(args.root + 'downstream_model/' + args.savedir)
                    net_path = args.root + 'downstream_model/' + args.savedir + '/Train_' +args.train_sensor +'_test_ ' +args.test_sensor +'_TOP_ ' +str(crite_list.index(crit_save ) +1 ) +'Net.pth'
                    torch.save(net.state_dict() ,net_path)
            if test_counter == 0:
                t.set_postfix_str \
                    ('Val_Acc:{:.2f}, Test_Acc:{:.2f}, TDR_val:{:.2f}, TDR_test:{:.2f}'.format(acc_val, acc_te, tdr_val, tdr_te))
            else:
                t.set_postfix_str \
                    ('TrLoss : {:.2f}, Val_Loss:{:.2f}, Test_Loss:{:.2f}'.format(loss_tr, loss_val, loss_te))
            t.update()

    # testing the performance of the 5 saved models on the whole testing set
    for i in range(1, 6):
        try:
            result = test(search=[switch[args.test_sensor], 'test'], sensor_te=switch[args.test_sensor], num=i)
            logger.info("TOP%d:\n \t\t\tACE:%5.2f TDR:%5.2f AUC:%5.2f EER:%5.2f" % (i, result[0]*100, result[1]*100, result[2]*100, result[3]*100))
        except:
            pass


def test(search, sensor_te, num):
    net = MobileNetV2()
    net.load_state_dict(torch.load(args.root + 'downstream_model/' + args.savedir + '/Train_' +args.train_sensor +'_test_ ' +args.test_sensor +'_TOP_ ' +str(num) +'Net.pth'))

    test_data = LivDetDataset(args.dataLabel_path, search)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_dataiter = iter(test_loader)

    net.cuda()
    net.eval()

    correct = torch.zeros(1).squeeze().cuda()
    result = []
    prob_list = []
    label_list = []
    with torch.no_grad():
        with tqdm(total=len(test_data.imgs), ncols=110, desc='Test:' + sensor_te) as t:
            inum = 1
            for img, l in test_dataiter:
                img = img.type(torch.FloatTensor).cuda()
                l = l.cuda().view(-1)
                out = net(img)
                out = F.softmax(out, dim=1)
                pred = torch.argmax(out, 1)
                correct += (pred == l).sum().float()
                acc = (correct / inum).cpu().detach().data.numpy()
                result.append([l.cpu().detach().data.numpy()[0], out.cpu().detach().data.numpy()[0, 1]])
                prob_list = np.append(prob_list, out.cpu().detach().data.numpy()[0, 1])
                label_list = np.append(label_list, l.cpu().detach().data.numpy()[0])
                inum += 1
                t.set_postfix_str('ACC={:^7.3f}'.format(acc))
                t.update()

    ace, tdr = ACE_TDR_Cal(result, rate=0.01)
    auc_score = roc_auc_score(label_list, prob_list)
    eer_score, threshold, _, _ = get_EER_states(prob_list, label_list)
    print('ACE : {:^4f}    TDR@FDR=1% : {:^4f}'.format(ace, tdr))
    return [ace, tdr, auc_score, eer_score]

if __name__ == "__main__":
    train()
