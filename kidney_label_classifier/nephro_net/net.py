
import torch
import torch.nn as nn
from torchvision import models

class Net(nn.Module):

    def __init__(self, task, mod = 'vgg'):
        super().__init__()
        self.task = task
        self.mod = mod

        task_d = {'view': 4, 'granular': 6, 'bladder': 2}
        # in features
        model_d = {'alexnet': 256, 'vgg': 512, 'resnet':2048, 'densenet' :1024, 'squeezenet':512,
                   'custom':0} # doesn't matter


        in_f = model_d[self.mod]
        out = task_d[self.task]

        # both vgg and alexnet have same final fc layer
        if self.mod == 'alexnet':
            print('alexnet')
            self.model = models.alexnet(pretrained = True)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.clf_on = nn.Linear(in_f, out)  # 256 if alexnet, 512 if vgg or resnet
        elif self.mod == 'vgg':
            print('vgg')
            self.model = models.vgg11_bn(pretrained=True)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.clf_on = nn.Linear(in_f, out)  # 256 if alexnet, 512 if vgg or resnet
        elif self.mod == 'resnet':
            print('resnet')
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(in_features = in_f, out_features = out, bias = True)
        elif self.mod == 'densenet':
            print('densenet')
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(in_features = in_f, out_features = out, bias = True)
        elif self.mod == 'squeezenet':
            print('densenet')
            self.model = models.squeezenet1_1(pretrained=True)
            self.model.classifier[1] = nn.Conv2d(512, out, kernel_size=(1,1), stride=(1,1))

        # input is 256x256
        # let's try a simple repeated cbr-maxpool with small filters (5)
        # so we repeat cbr-max 4 times, global average pool into classification

        # 3 x 256 x 256
        # outputfilter w/h = (W−F+2P) / S) + 1, ((256 - 5 + 0)/1) + 1
        # # p = (f - 1)/2 for stride = 1 for p for same padding
        # x = torch.randn([1, 3, 256, 256 ])
        # x1 = (maxp(relu(bn64(conv64(x)))))
        # x2 = (maxp(relu(bn128(conv128(x1)))))
        # x3 = (maxp(relu(bn256(conv256(x2)))))
        # x4 = (maxp(relu(bn512(conv512(x3)))))
        # x5 = classifier(gap(x4).view(x4.size(0), -1))
        # p = 2 for same padding, formula?

        self.conv64 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 1, padding = 0)
        self.conv128 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 7, stride = 1, padding = 0)
        self.conv256 = nn.Conv2d(in_channels=128, out_channels = 256, kernel_size=7, stride=1, padding=0)
        self.conv512 = nn.Conv2d(in_channels=256, out_channels = 512, kernel_size=7, stride=1, padding=0)
        # outputs filters of w - f + 2p/s + 1, or 251

        self.bn64 = nn.BatchNorm2d(64)  #
        self.bn128 = nn.BatchNorm2d(128)  #
        self.bn256 = nn.BatchNorm2d(256)  #
        self.bn512 = nn.BatchNorm2d(512)  #
        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size = 3, stride = 2)

        # first conv-batch-relu pass gives 32 x 127 x 127

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features = 512, out_features = out, bias = True)

    def forward(self, x):
        just_forward = ('resnet', 'squeezenet', 'densenet')

        if self.mod in just_forward:
            x = self.model(x)
        elif self.mod == 'custom':
            x = self.maxp(self.relu(self.bn64(self.conv64(x))))
            # print(x.shape)
            x = self.maxp(self.relu(self.bn128(self.conv128(x))))
            # print(x.shape)
            x = self.maxp(self.relu(self.bn256(self.conv256(x))))
            # print(x.shape)
            x = self.maxp(self.relu(self.bn512(self.conv512(x))))
            # print(x.shape)
            x = self.gap(x).view(x.size(0), -1) # to 'unroll'
            # print(x.shape)
            x = self.classifier(x)
            # print(x.shape)
        else: #alexnet, vgg, feature extractor, with adaptive pooling
            x = self.model.features(x)
        # print(x.shape)#, mb x256 x 7 x 7 // 512 x 8 x 8
            x = self.gap(x).view(x.size(0), -1) # ADAPTIVE POOL, 'collapses' into mb x 256 x 1, then mb x 256
        # print(x.shape) # mb x 256
        # x = torch.max(x, 0, keepdim = True)[0] # max of each column
        # print(x.shape)
            x = self.clf_on(x)
        # print(x.shape)
        return x


# misc stuff

# # from torchsummary import summary
# model = Net(task = 'bladder', mod = 'custom')
# model2 = Net(task = 'bladder', mod = 'resnet')
# model3 = Net(task = 'bladder', mod = 'alexnet')
# # x = torch.randn([1, 3, 256, 256 ])
# # model.forward(x)
# # #
# # model.forward(x)
# summary(model, (3, 256, 256), batch_size = 1) # custom, 7x7 3 CBR layers
# summary(model2, (3, 256, 256), batch_size = 1) # resnet
# summary(model3, (3, 256, 256), batch_size = 1) # alexnet, just the feature extraction layers though