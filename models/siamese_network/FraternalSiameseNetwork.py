from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SiamNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2):
        super(SiamNet, self).__init__()

        self.num_inputs = num_inputs
        # ********************** ITEM 0 ********************* #
        self.conv_0 = nn.Sequential()
        self.conv_0.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv_0.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv_0.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv_0.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv_0.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv_0.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv_0.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv_0.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv_0.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv_0.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv_0.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv_0.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv_0.add_module('relu3_s1', nn.ReLU(inplace=True))
        self.conv_0.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv_0.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv_0.add_module('relu4_s1', nn.ReLU(inplace=True))
        self.conv_0.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv_0.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv_0.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv_0.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6_0 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6_0.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.fc6_0.add_module('batch6_s1', nn.BatchNorm2d(1024))

        self.fc6b_0 = nn.Sequential()
        self.fc6b_0.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        # self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b_0.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b_0.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c_0 = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c_0.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c_0.add_module('relu7', nn.ReLU(inplace=True))
        self.fc6c_0.add_module('drop7', nn.Dropout(p=0.5))

        # ********************** ITEM 1 ********************* #
        self.conv_1 = nn.Sequential()
        self.conv_1.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv_1.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv_1.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv_1.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv_1.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv_1.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv_1.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv_1.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv_1.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv_1.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv_1.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv_1.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv_1.add_module('relu3_s1', nn.ReLU(inplace=True))
        self.conv_1.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv_1.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv_1.add_module('relu4_s1', nn.ReLU(inplace=True))
        self.conv_1.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv_1.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv_1.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv_1.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6_1 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6_1.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.fc6_1.add_module('batch6_s1', nn.BatchNorm2d(1024))

        self.fc6b_1 = nn.Sequential()
        self.fc6b_1.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        # self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b_1.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b_1.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c_1 = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c_1.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c_1.add_module('relu7', nn.ReLU(inplace=True))
        self.fc6c_1.add_module('drop7', nn.Dropout(p=0.5))


        # ***************** LAYERS THAT TAKE IN ALL INPUTS **************** #
        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, 4096))
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

        curr_x_0 = torch.unsqueeze(x[0], 1)
        curr_x_0 = curr_x_0.expand(-1, 3, -1, -1)
        if torch.cuda.is_available():
            input_0 = torch.cuda.FloatTensor(curr_x_0.to(device))
        else:
            input_0 = torch.FloatTensor(curr_x_0.to(device))
        z_0 = self.conv_0(input_0)
        z_0 = self.fc6_0(z_0)
        z_0 = self.fc6b_0(z_0)
        z_0 = z_0.view([B, 1, -1])
        z_0 = self.fc6c_0(z_0)
        z_0 = z_0.view([B, 1, -1])
        x_list.append(z_0)

        curr_x_1 = torch.unsqueeze(x[1], 1)
        curr_x_1 = curr_x_1.expand(-1, 3, -1, -1)
        if torch.cuda.is_available():
            input_1 = torch.cuda.FloatTensor(curr_x_1.to(device))
        else:
            input_1 = torch.FloatTensor(curr_x_1.to(device))
        z_1 = self.conv_0(input_1)
        z_1 = self.fc6_0(z_1)
        z_1 = self.fc6b_0(z_1)
        z_1 = z_1.view([B, 1, -1])
        z_1 = self.fc6c_0(z_1)
        z_1 = z_1.view([B, 1, -1])
        x_list.append(z_1)

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
