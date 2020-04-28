from torch import nn
import torch
#softmax = torch.nn.Softmax(dim=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_views=2, dropout_rate=0.5, three_channel=False):
        super(CNN, self).__init__()
        self.num_views = num_views
        self.three_channel = three_channel
        
        self.conv1 = nn.Sequential()
        if three_channel:
            self.conv1.add_module('conv',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        else:
            self.conv1.add_module('conv',nn.Conv2d(1, 96, kernel_size=11, stride=2, padding=0))
        self.conv1.add_module('batch', nn.BatchNorm2d(96))
        self.conv1.add_module('relu',nn.ReLU(inplace=True))
        self.conv1.add_module('pool1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv1.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv2.add_module('batch', nn.BatchNorm2d(256))
        self.conv2.add_module('relu',nn.ReLU(inplace=True))
        self.conv2.add_module('pool',nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.conv2b= nn.Sequential()
        self.conv2b.add_module('conv', nn.Conv2d(256, 256, kernel_size=2, padding=1, stride=1))
        self.conv2b.add_module('batch', nn.BatchNorm2d(256))
        self.conv2b.add_module('relu',nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv3.add_module('batch', nn.BatchNorm2d(384))
        self.conv3.add_module('relu',nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv4.add_module('batch', nn.BatchNorm2d(384))
        self.conv4.add_module('relu',nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('conv',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv5.add_module('batch', nn.BatchNorm2d(256))
        self.conv5.add_module('relu',nn.ReLU(inplace=True))
        self.conv5.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv6 = nn.Sequential()
        self.conv6.add_module('conv', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.conv6.add_module('batch', nn.BatchNorm2d(1024))
        self.conv6.add_module('relu', nn.ReLU(inplace=True))
        self.conv6.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=1))

        self.conv7 = nn.Sequential()
        self.conv7.add_module('conv', nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.conv7.add_module('batch', nn.BatchNorm2d(256))
        self.conv7.add_module('relu', nn.ReLU(inplace=True))
        self.conv7.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc1 = nn.Sequential()
        self.fc1.add_module('linear', nn.Linear(256*7*7, 512))
        self.fc1.add_module('relu', nn.ReLU(inplace=True))
        self.fc1.add_module('drop', nn.Dropout(p=dropout_rate))

        self.fc2_singleside = nn.Sequential()
        self.fc2_singleside.add_module('linear', nn.Linear(self.num_views * 512, 512))
        self.fc2_singleside.add_module('relu', nn.ReLU(inplace=True))

        self.fc2_bothsides = nn.Sequential()
        self.fc2_bothsides.add_module('linear', nn.Linear(self.num_views * 512 * 2, 512))
        self.fc2_bothsides.add_module('relu', nn.ReLU(inplace=True))

        
        self.classifier_surgery = nn.Sequential()
        self.classifier_surgery.add_module('linear', nn.Linear(512, 2))

        self.classifier_reflux = nn.Sequential()
        self.classifier_reflux.add_module('linear', nn.Linear(512, 2))

        self.classifier_function = nn.Sequential()
        self.classifier_function.add_module('linear', nn.Linear(512, 2))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, label_type):
        if self.num_views == 1:
            x = x.unsqueeze(1)
        
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        xbar_list = []
        
        for i in range(T):
            curr_x = torch.unsqueeze(x[i], 1)
            if self.three_channel:
                curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))


            out1 = self.conv1(input)
            out2 = self.conv2(out1)
            out2b = self.conv2b(out2)
            out3 = self.conv3(out2b)
            out4 = self.conv4(out3)
            out5 = self.conv5(out4)
            out6 = self.conv6(out5)
            out7 = self.conv7(out6).view([B, 1, -1])
            out8 = self.fc1(out7).view([B, 1, -1])
            xbar_list.append(out8)

        xbar_concat = torch.cat(xbar_list, 1)
        if label_type != 'f':
            out9 = self.fc2_singleside(xbar_concat.view(B, -1))
        else:
            out9 = self.fc2_bothsides(xbar_concat.view(B, -1))

        
        if label_type == 's':
            pred = self.classifier_surgery(out9)
        elif label_type == 'r':
            pred = self.classifier_reflux(out9)
        elif label_type == 'f':
            pred = self.classifier_function(out9)
        
        return pred

