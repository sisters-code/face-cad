import os
from logger import Logger
import torch
import torch.utils.data
from opts import opts
from model import getModel
import ref
import sys
import numpy as np

opt = opts().parse()

from datasets.face_CAD import face_CAD as Dataset
from train import train, val, test

def main():
    logger = Logger(opt.logDir)

    # write reference and related information about this experiment into 'opt.txt'
    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    refs = dict((name, getattr(ref, name)) for name in dir(ref)
                if not name.startswith('_'))
    if not os.path.exists(opt.logDir):
        os.makedirs(opt.logDir)
    file_name = os.path.join(opt.logDir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('==> Cmd:\n')
        opt_file.write(str(sys.argv))
        opt_file.write('\n==> Opt:\n')
        for k, v in sorted(args.items()):
            opt_file.write('  %s: %s\n' % (str(k), str(v)))
        opt_file.write('==> Ref:\n')
        for k, v in sorted(refs.items()):
            opt_file.write('  %s: %s\n' % (str(k), str(v)))

    #prepare model and optimizer
    model, optimizer = getModel(opt)
    #use MSEloss as criterion
    criterion = torch.nn.MSELoss()

    #move the model, optimizer and criterion to the GPU
    if opt.GPU > -1:
        print('Using GPU', opt.GPU)
        model = model.cuda(opt.GPU)
        criterion = criterion.cuda(opt.GPU)
        #optimizer = optimizer.cuda(opt.GPU)Í

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.trainBatch,
        shuffle=True,
        num_workers=int(opt.nThreads)
    )

    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=True if opt.Debug else False,
        num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'test'),
        batch_size=1,
        # shuffle=True if opt.Debug else False,
        shuffle=False,
        num_workers=1
    )

    for epoch in range(1, opt.nEpochs + 1):
        log_dict_train, _ = train(epoch, opt, train_loader, model, logger.tensorboard, optimizer)
        if epoch % opt.valIntervals == 0:
            log_dict_val, _ = val(epoch, opt, val_loader, model, logger.tensorboard)
        if epoch % opt.dropLR == 0:
            lr = opt.LR * (0.1 ** (epoch // opt.dropLR))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if log_dict_train['Acc'] >= 99:
            break
    path = os.path.join(ref.expDir, opt.expID, 'best_val_auc.checkpoint')
    net_load = torch.load(path)
    model.load_state_dict(net_load['state_dict'])
    test( epoch, opt, test_loader, model, logger.tensorboard)
    logger.f.write('best val auc: {}, test auc: {}'.format(opt.best_val_auc, opt.best_test_auc))
    logger.f.close()

'''
        if epoch == 15:
            if opt.arch.startswith('resnet'):
                param_name_finetune = ['fc', 'layer4']
            elif opt.arch.startswith('vgg'):
                param_name_finetune = ['classifier.0', 'classifier.3', 'classifier.6', 'features.31', 'features.34',
                                       'features.35', 'features.37', 'features.38', 'features.40', 'features.41', ]
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


    torch.save(model.cpu(), os.path.join(opt.logDir, 'model_cpu.pth'))
'''

if __name__ == '__main__':
    main()
