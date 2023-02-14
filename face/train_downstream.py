import sys
sys.path.append('../')
import argparse
from face.utils.utils import get_dataset, AverageMeter
from face.utils.evaluate import accuracy, eval, re_eval
import random
import numpy as np
import torch
from face.models.resnet import Resnet18
from tqdm import *
import logging
import torch.nn as nn
import os

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root",type=str, default='./', help='Root path')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=5)
parser.add_argument("--train_data", type=str, default='om', help='Training data (om/ci)')
parser.add_argument("--test_data", type=str, default='ci', help='Testing data (om/ci)')
parser.add_argument("--opt", type=str, default='SGD', help='Choose optimizer(SGD/Adam)')
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--Epoch_num", type=int, default=90, help='Training epoch number')
parser.add_argument("--eval_epoch", type=int, default=1, help='Evaluate frequency')
parser.add_argument("--savedir", type=str, help='Path to save the models')
parser.add_argument("--crite", type=str, default='eer')
parser.add_argument("--model", type=str, default='load', help='Type of model, choose the model from imagenet or ifom_pretext.py(imagenet/load)')
parser.add_argument("--loadModelPath", type=str, help='The path of the model obtained from ifom_pretext.py')
args = parser.parse_args()

###################################################
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
####################################################

logger = logging.getLogger("mainModule")
logger.setLevel(level=logging.DEBUG)
log_dir = args.root + 'downstream_log/' + args.savedir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = log_dir + '/Train_' + args.train_data + '_test_' + args.test_data + '.txt'
handler = logging.FileHandler(filename=log_path, mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info("Log path:" + log_path)
logger.info("root:" + args.root)
logger.info("batch_size:{}".format(args.batch_size))
logger.info("seed:{}".format(args.seed))
logger.info("Train_data:{}".format(args.train_data))
logger.info("Test_data:{}".format(args.test_data))
logger.info("opt:{}".format(args.opt))
logger.info("learning rate:{:^.4f}".format(args.learning_rate))
logger.info("momentum:{:^.4f}".format(args.momentum))
logger.info("weight_decay:{:^.4f}".format(args.weight_decay))
logger.info("Epoch_num:{}".format(args.Epoch_num))
logger.info("eval_epoch:{}".format(args.eval_epoch))
logger.info("savedir:{}".format(args.savedir))
logger.info("model type:{}".format(args.model))
logger.info("load model path:{}".format(args.loadModelPath))


def inin():
    try:
        train_data_loader, test_data_loader = get_dataset('./labels', args.train_data, 1, args.test_data, 1, args.batch_size)
    except FileNotFoundError:
        print("Error: Please download the dataset.")
        sys.exit()
    net = Resnet18()
    if args.model == 'load':
        model_dict = net.state_dict()
        pretrained_dict = torch.load(args.loadModelPath)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    elif args.model == 'imagenet':
        try:
            model_dict = net.state_dict()
            logger.info("load imagenet model path:{}".format('./pretrained_model/resnet18-5c106cde.pth'))
            pretrained_dict = torch.load('./pretrained_model/resnet18-5c106cde.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        except FileNotFoundError:
            print("Error: The pretrained model will be made publicly available soon.")
            sys.exit()
    return net, train_data_loader, test_data_loader

def train():
    net, train_loader, test_loader = inin()
    save_num = 0
    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_EER = 1.0
    best_model_AUC = 0.0
    best_model_ACE = 1.0
    best_model_TDR = 0.0
    # loss,top1 accuracy, eer, hter, auc, threshold, acc, ace, tdr
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0, 0]

    logger.info('****************************************** start training target model! *************************************\n')
    logger.info(
        '--------|--------------------- VALID --------------------|--- classifier ---|------------- Current Best -------------|\n')
    logger.info(
        '  epoch  |   loss   top-1   EER     AUC    ACE    TDR    |   loss   top-1   |   top-1   EER     AUC    ACE    TDR    |\n')
    logger.info(
        '---------------------------------------------------------------------------------------------------------------------|\n')

    net = net.cuda()
    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)


    for e in range(args.Epoch_num):
        t = tqdm(train_loader)
        t.set_description("Epoch [{}/{}]".format(e +1 ,args.Epoch_num))
        for b, (imgs, labels, _) in enumerate(t):
            net.train()
            imgs = imgs.cuda()
            labels = labels.cuda().view(-1)
            out = net(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_classifier.update(loss.item())
            acc = accuracy(out.narrow(0, 0, imgs.size(0)), labels, topk=(1,))
            classifer_top1.update(acc[0])

        if ((e+1) % args.eval_epoch == 0):
            valid_args = eval(test_loader, net)
            if args.crite == 'eer':
                is_best = valid_args[2] <= best_model_EER
            elif args.crite == 'ace':
                is_best = valid_args[7] <= best_model_ACE
            elif args.crite == 'tdr':
                is_best = valid_args[8] >= best_model_TDR

            if (is_best):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]
                best_model_EER = valid_args[2]
                best_model_ACE = valid_args[7]
                best_model_TDR = valid_args[8]

            logger.info(
                '  %3d  |  %5.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  |'
                % (
                e+1,
                valid_args[0], valid_args[6] * 100, valid_args[2] * 100, valid_args[4] * 100, valid_args[7] *100, valid_args[8] *100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC * 100), float(best_model_EER * 100), float(best_model_AUC * 100), float(best_model_ACE * 100), float(best_model_TDR * 100)))
            if is_best:
                if not os.path.exists(args.root + 'downstream_model/' + args.savedir):
                    os.makedirs(args.root + 'downstream_model/' + args.savedir)
                save_num = save_num + 1
                save_path = args.root + 'downstream_model/' + args.savedir + '/Train_' + args.train_data + '_test_' + args.test_data + '_' + str(save_num) + '.pth'
                torch.save(net.state_dict(), save_path)


    result = re_eval(test_loader, args.root + 'downstream_model/' + args.savedir + '/Train_' + args.train_data + '_test_' + args.test_data + '_', save_num)
    logger.info('EER:%.2f' % (result[2] * 100))
    logger.info('AUC:%.2f' % (result[4] * 100))
    logger.info('ACE:%.2f' % (result[7] * 100))
    logger.info('TDR:%.2f' % (result[8] * 100))


if __name__ == "__main__":
    train()
