from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiamNet(nn.Module):
    def __init__(self, classes=2):
        super(SiamNet, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv1.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv1.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv1.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv1.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv2.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv3.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv3.add_module('relu3_s1',nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv4.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv4.add_module('relu4_s1',nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv5.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_s1',nn.ReLU(inplace=True))
        # self.conv5.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv5.add_module('pool5_s1', nn.MaxPool2d(kernel_size=2, stride=2))

        # *************************** changed layers *********************** #

        self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        # self.fc6.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # self.fc6b = nn.Sequential()
        # # self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        # self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 96, kernel_size=1, stride=1))
        # self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(96))
        # self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        # self.fc6b.add_module('upsample', nn.Upsample(scale_factor=4))
        #
        # # self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # ***********************************************************************#
        # U-NET #
        # self.uconnect= nn.Sequential()
        # self.uconnect.add_module('conv_sc', nn.Conv2d(2*96, 256, kernel_size=3, stride=2))
        # self.uconnect.add_module('batch_sc', nn.BatchNorm2d(256))
        # self.uconnect.add_module('relu_sc', nn.ReLU(inplace=True))
        # self.uconnect.add_module('pool_sc', nn.MaxPool2d(kernel_size=3, stride=2))
        # ***********************************************************************#

        # ***********************************************************************#
        # U-NET #
        self.uconnect1= nn.Sequential()
        self.uconnect1.add_module('conv', nn.Conv2d(1024+256, 256, kernel_size=2, stride=2))
        self.uconnect1.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect1.add_module('relu', nn.ReLU(inplace=True))
        self.uconnect1.add_module('upsample', nn.Upsample(scale_factor=2))  # 256 * 30 * 30

        self.uconnect2= nn.Sequential()
        self.uconnect2.add_module('conv', nn.Conv2d(384+256, 256, kernel_size=3, stride=2, padding=1))
        self.uconnect2.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect2.add_module('relu', nn.ReLU(inplace=True))
        self.uconnect2.add_module('upsample', nn.Upsample(scale_factor=2)) # 256 * 30 * 30

        self.uconnect3= nn.Sequential()
        self.uconnect3.add_module('conv', nn.Conv2d(384+256, 256, kernel_size=3, stride=2, padding=1))
        self.uconnect3.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect3.add_module('relu', nn.ReLU(inplace=True))
        self.uconnect3.add_module('upsample', nn.Upsample(scale_factor=2)) # 256 * 30 * 30

        self.uconnect4= nn.Sequential()
        self.uconnect4.add_module('conv', nn.Conv2d(2*256, 256, kernel_size=3, stride=2, padding=1))
        self.uconnect4.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect4.add_module('relu', nn.ReLU(inplace=True))
        self.uconnect4.add_module('upsample', nn.Upsample(scale_factor=4)) # 256 * 60 * 60

        self.uconnect5 = nn.Sequential()
        self.uconnect5.add_module('conv', nn.Conv2d(96+256, 256, kernel_size=5, stride=2))
        self.uconnect5.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect5.add_module('relu', nn.ReLU(inplace=True))
        self.uconnect5.add_module('upsample', nn.Upsample(scale_factor=2))
        # ***********************************************************************#

        self.fc6c = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c.add_module('fc7', nn.Linear(256*14*14, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        self.fc6c.add_module('drop7', nn.Dropout(p=0.5))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(2 * 512, 4096))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7_new.add_module('drop7', nn.Dropout(p=0.5))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(4096, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []

        for i in range(2):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))


            out1 = self.conv1(input)
            out2 = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            print(out4.shape)
            out5 = self.conv5(out4)
            print(out5.shape)
            out6 = self.fc6(out5)
            print(out6.shape)

            unet1 = self.uconnect1(torch.cat((out5, out6), dim=1))
            unet2 = self.uconnect2(torch.cat((out4, unet1), dim=1))
            unet3 = self.uconnect3(torch.cat((out3, unet2), dim=1))
            unet4 = self.uconnect4(torch.cat((out2, unet3), dim=1))

            unet4 = unet4.expand(-1, -1, 61, 61)
            unet5 = self.uconnect4(torch.cat((out1, unet4), dim=1))

            z = unet5.view([B, 1, -1])
            print(z.shape)
            z = self.fc6c(z)
            print(z.shape)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        return pred

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)

            div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x
