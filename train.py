import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import AverageMeter
from utils.eval import AccCls
import cv2
import ref
from progress.bar import Bar
from datetime import datetime
import matplotlib.pyplot as plt
from model import saveModel

plt.switch_backend('Agg')
from sklearn.metrics import roc_curve, auc
from PIL import Image
import os
import csv

def train(epoch, opt, train_loader, model, tsbd ,optimizer):
    return step('train', epoch, opt, train_loader, model, tsbd, optimizer)

def val(epoch, opt, val_loader, model, tsbd):
    return step('val', epoch, opt, val_loader, model, tsbd)

def test( epoch, opt, test_loader, model, tsbd):
    return step('test', epoch, opt, test_loader, model, tsbd)

def step(split, epoch, opt, dataLoader, model, tensorboard=None, optimizer=None):
    acc = 0.
    count = 0
    label_list = []
    scores_list = []
    if split == 'train':
        model.train()
    else:
        # Validate the model
        print("{} Start {}dation".format(datetime.now(), split))
        score_path = os.path.join(opt.logDir, "{}/WrongPred_score_{}_epoch_{}.txt".format(opt.logDir, split, epoch))
        scoretxt = open(score_path, 'wt')
        model.eval()

    Loss, Acc = AverageMeter(), AverageMeter()
    nIters = len(dataLoader)
    bar = Bar('{}'.format(opt.expID), max=nIters)

    for i, (input, gtCls, imPath) in enumerate(dataLoader):
        input_var = torch.autograd.Variable(input.cuda(opt.GPU, non_blocking=True)).float().cuda(opt.GPU)
        target_var = torch.autograd.Variable(gtCls.view(-1)).long().cuda(opt.GPU)
        output = model(input_var)

        numBins = opt.numBins
        loss = torch.nn.CrossEntropyLoss(ignore_index=numBins).cuda(opt.GPU)(output.view(-1, numBins), target_var)
        Acc.update(AccCls(output.data.cpu(), gtCls))
        Loss.update(loss.item(), input.size(0))

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tensorboard.add_scalar("train/Loss", loss, (epoch - 1) * nIters + i)
            tensorboard.add_scalar("train/Accuracy", Acc.avg, (epoch - 1) * nIters + i)
        else:
            for idx in range(len(gtCls)):
                score = F.softmax(output[idx], dim=0)[1].data.cpu().numpy()
                true_score = F.softmax(output[idx], dim=0)[gtCls[idx]].data.cpu().numpy()
                label = gtCls.numpy()[idx]

                if int(score + 0.5) == label:
                    acc += 1
                else:
                    scoretxt.write(imPath[idx] + ': label: {}+score: {}\n'.format(label, true_score))
                count += 1
                label_list.append(label)
                scores_list.append(score)
            tensorboard.add_scalar("{}/Loss".format(split), loss, (epoch - 1) * nIters + i)
            tensorboard.add_scalar("{}/Accuracy".format(split), Acc.avg, (epoch - 1) * nIters + i)


        Bar.suffix = '{split:5} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f}'.format(
            epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split=split)
        bar.next()

    bar.finish()

    if split != 'train':
        AUC = compute_AUC(label_list, scores_list, opt, split, epoch)
        tensorboard.add_scalar("{}/AUC".format(split), AUC, (epoch - 1) * nIters + i)
        acc /= count
        print("{} {} Accuracy = {:.4f}".format(datetime.now(), split, acc))
        scoretxt.write("{} {} Accuracy = {:.4f}, AUC = {:.4f}\n".format(datetime.now(), split, acc, AUC))
        scoretxt.close()

        #in this validation progress, the validation AUC has been added into the val_auc_list
        if split == 'val':
            print('val auc list: {}'.format(opt.val_auc_list))
            if AUC == np.max(opt.val_auc_list):
                opt.best_val_auc = AUC
                saveModel(os.path.join(opt.logDir, 'best_val_auc.checkpoint'), model)
                print('best val auc: {}'.format(AUC))
        elif split == 'test':
            opt.best_test_auc = AUC
    return {'Loss': Loss.avg, 'Acc': Acc.avg}, label_list

def finetune(opt, model):
        if opt.arch.startswith('resnet'):
            param_name_finetune = ['fc', 'layer4']
        elif opt.arch.startswith('vgg'):
            param_name_finetune = ['classifier.0', 'classifier.3', 'classifier.6', 'features.31', 'features.34',
                                   'features.35', 'features.37', 'features.38', 'features.40', 'features.41', ]
        else:
            raise Exception('resnet or vgg')
        param_finetune = []
        for name, param in model.named_parameters():
            finetune_flag = False
            for name_tgt in param_name_finetune:
                if name_tgt in name:
                    finetune_flag = True
                    break
            if finetune_flag:
                param_finetune.append(param)
            else:
                param.requires_grad = False
        optimizer = torch.optim.SGD(param_finetune, opt.LR, momentum=0.9, weight_decay=1e-4)
        return optimizer


def compute_AUC(label_list, scores_list, opt, split, epoch):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(np.array(label_list), np.array(scores_list), pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("{} AUC: {}".format(split, roc_auc))
    if split == 'val':
        opt.val_auc_list.append(roc_auc)
    elif split == 'test':
        opt.test_auc_list.append(roc_auc)
    plt.title('ECG {}'.format(split))
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("{}/AUC_{}_epoch_{}.jpg".format(opt.logDir, split, epoch))
    plt.close()

    return roc_auc


