import torchvision
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RevisedResNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2):
        super(RevisedResNet, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.resnet18(pretrained=False)
        self.old_arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.old_arch.fc = nn.Linear(512, 206, bias=True)

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 206, 206, bias=True),
                                         nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(206, classes, bias=True),
                                       nn.Sigmoid())

    def forward(self, x):
        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

                res_out = self.old_arch(input)
                x_list.append(res_out)

        x = torch.cat(x_list, 1)
        x = x.view(B, -1)
        x = self.combo_layer(x)
        pred = self.out_layer(x)

        return pred
