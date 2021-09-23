import torchvision.models as models
import ref
import torch
import torch.nn as nn
import os
'''
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.resnet_layer(x)

        x = x.view(x.size(0), -1)

        return x
'''

# Re-init optimizer
def getModel(opt):
    print("=> using pre-trained model '{}'".format(opt.arch))
    model = models.__dict__[opt.arch](pretrained=True)
    if opt.arch.startswith('resnet'):
        # model.avgpool = nn.AvgPool2d(4, stride=1) # image size=128 : 4 # image size=384 : 12
        model.avgpool = nn.AvgPool2d(8, stride=1)
        if '18' in opt.arch:
            model.fc = nn.Linear(512 * 1, opt.numBins)
        else:
            model.fc = nn.Linear(512 * 4, opt.numBins)
        model.conv1 = nn.Conv2d(opt.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        ###
        # model = FeatureExtractor(model)#
    if opt.arch.startswith('densenet'):
        if '161' in opt.arch:
            model.classifier = nn.Linear(2208, opt.numOutput)
        elif '201' in opt.arch:
            model.classifier = nn.Linear(1920, opt.numOutput)
        else:
            model.classifier = nn.Linear(1024, opt.numOutput)
    if opt.arch.startswith('vgg'):
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(4096, opt.numOutput))
        model.classifier = nn.Sequential(*feature_model)
        model.features[0] = nn.Conv2d(12, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    optimizer = torch.optim.SGD(model.parameters(), opt.LR, momentum=0.9, weight_decay=1e-4)

    return model, optimizer

def load_model(opt,model):
    #将之前已经训练好的模型权重写入网络中
    if opt.test != '':
        checkpoint = os.path.join(ref.expDir, opt.expID, 'best_val_auc.checkpoint')
        print("=> loading model '{}'".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()

        model_dict = model.state_dict()  #
        # filter out unnecessary params
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}  #
        # update state dict
        model_dict.update(filtered_state_dict)  #
        model.load_state_dict(model_dict)

        # model.load_state_dict(state_dict)


def saveModel(path, model, optimizer=None):
    if optimizer is None:
        torch.save({'state_dict': model.state_dict()}, path)
    else:
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, path)
