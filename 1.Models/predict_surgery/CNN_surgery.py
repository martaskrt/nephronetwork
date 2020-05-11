from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_inputs=2, classes=2, dropout_rate=0.5, output_dim=256):
        super(CNN, self).__init__()
        self.num_inputs = num_inputs
        self.output_dim=output_dim
        
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv_s1',nn.Conv2d(1, 96, kernel_size=11, stride=2, padding=0))
        self.conv1.add_module('batch_s1', nn.BatchNorm2d(96))
        self.conv1.add_module('relu_s1',nn.ReLU(inplace=True))
        self.conv1.add_module('pool_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv1.add_module('pool_s2', nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv2.add_module('batch_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu_s1',nn.ReLU(inplace=True))
        self.conv2.add_module('pool_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv_s1', nn.Conv2d(256, 256, kernel_size=2, padding=1, stride=1))
        self.conv3.add_module('batch_s1', nn.BatchNorm2d(256))
        self.conv3.add_module('relu_s1',nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv4.add_module('batch_s1', nn.BatchNorm2d(384))
        self.conv4.add_module('relu_s1',nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('conv_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv5.add_module('batch_s1', nn.BatchNorm2d(384))
        self.conv5.add_module('relu_s1',nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential()
        self.conv6.add_module('conv_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv6.add_module('batch_s1', nn.BatchNorm2d(256))
        self.conv6.add_module('relu_s1',nn.ReLU(inplace=True))
        self.conv6.add_module('pool_s1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv7 = nn.Sequential()
        self.conv7.add_module('conv_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.conv7.add_module('batch_s1', nn.BatchNorm2d(1024))
        self.conv7.add_module('relu_s1', nn.ReLU(inplace=True))
        self.conv7.add_module('pool_s1', nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.conv8 = nn.Sequential()
        self.conv8.add_module('conv', nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.conv8.add_module('batch', nn.BatchNorm2d(256))
        self.conv8.add_module('relu', nn.ReLU(inplace=True))
        self.conv8.add_module('pool_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc1 = nn.Sequential()
        self.fc1.add_module('lin', nn.Linear(256*7*7, 512))
        self.fc1.add_module('relu', nn.ReLU(inplace=True))
        self.fc1.add_module('drop', nn.Dropout(p=dropout_rate))

        self.fc2 = nn.Sequential()
        self.fc2.add_module('lin', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc2.add_module('relu', nn.ReLU(inplace=True))

        self.classifier = nn.Sequential()
        self.classifier.add_module('lin', nn.Linear(self.output_dim, classes))

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
        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []

        for i in range(self.num_inputs):
            curr_x = torch.unsqueeze(x[i], 1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

            out1 = self.conv1(input)
            out2 = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            out5 = self.conv5(out4)
            out6 = self.conv6(out5)
            out7 = self.conv7(out6)
            out8 = self.conv8(out7)

            flattened_out8 = out8.view([B, 1, -1])
            out9 = self.fc1(flattened_out8)
            x_list.append(out9.view([B, 1, -1]))

        x = torch.cat(x_list, 1)
        x = self.fc2(x.view(B, -1))
        pred = self.classifier(x)

        return pred

