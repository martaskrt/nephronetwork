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
        # self.old_arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.old_arch.fc = nn.Linear(512, 256, bias=True)

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))

    def forward(self, x):
        # print("x size: ")
        # print(x.size())

        # print("num_inputs: " + str(self.num_inputs))
        if self.num_inputs == 1:
            x = x.unsqueeze(1)

        # print("x size: ")
        # print(x.size())

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            # print("curr_x.size(): ")
            # print(curr_x.size())
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            # curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

            res_out = self.old_arch(input)
            x_list.append(res_out)


        #print(x_list)
        comb_out = torch.cat(x_list, 1)
        comb_out = comb_out.view(B, -1)
        comb_out2 = self.combo_layer(comb_out)
        pred = self.out_layer(comb_out2)

        return pred


class RevisedResNet50(nn.Module):
    def __init__(self, classes=2, num_inputs=2):
        super(RevisedResNet50, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.resnet50(pretrained=False)
        #self.old_arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.old_arch.fc = nn.Linear(2048, 256, bias=True)

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))

    def forward(self, x):

        if self.num_inputs == 1:
            x = x.unsqueeze(1)

        # print("x size: ")
        # print(x.size())

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            # print("curr_x.size(): ")
            # print(curr_x.size())
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            # curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

            res_out = self.old_arch(input)

            # print("res_out.size():")
            # print(res_out.size())

            x_list.append(res_out)

        #print(x_list)
        comb_out = torch.cat(x_list, 1)

        # print("comb_out.size():")
        # print(comb_out.size())

        comb_out = comb_out.view(B, -1)

        # print("comb_out.view(B,-1).size():")
        # print(comb_out.size())

        comb_out2 = self.combo_layer(comb_out)

        # print("comb_out2.size():")
        # print(comb_out2.size())

        pred = self.out_layer(comb_out2)

        # print("pred.size():")
        # print(pred.size())

        return pred

class RevisedVGG(nn.Module):
    def __init__(self, classes=2, num_inputs=2):
        super(RevisedVGG, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.vgg16(pretrained=False)

        self.first_seq = nn.Sequential(*list(self.old_arch.children())[0][:]) ## Remove input conv and replace with my own

        self.avg_pool = list(self.old_arch.children())[1]
        self.final_seq = nn.Sequential(*list(self.old_arch.children())[2][:-4],
                                       nn.Linear(in_features=4096, out_features=4096), ## potentially make this map 4096 to 4096 like orig VGG
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(in_features=4096, out_features=256))

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))


    def forward(self,x):


        if self.num_inputs == 1:
            x = x.unsqueeze(1)

        # print("x size: ")
        # print(x.size())

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            # print("curr_x.size(): ")
            # print(curr_x.size())
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            # curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

            out1 = self.first_seq(input)

            # print("out1.size():")
            # print(out1.size())

            out2 = self.avg_pool(out1)

            # print("out2.size():")
            # print(out2.size())

            out2_flat = out2.view(B, -1)

            # print("out2_flat.size():")
            # print(out2_flat.size())

            out3 = self.final_seq(out2_flat)
            x_list.append(out3)

        comb_out = torch.cat(x_list, 1)

        # print("comb_out.size():")
        # print(comb_out.size())

        #comb_out = comb_out.view(B, -1)
        comb_out2 = self.combo_layer(comb_out)
        pred = self.out_layer(comb_out2)

        return pred

## UPDATE AND TEST

class RevisedVGG_bn(nn.Module):
    def __init__(self, classes=2, num_inputs=2):
        super(RevisedVGG_bn, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.vgg16_bn(pretrained=False)

        self.first_seq = nn.Sequential(*list(self.old_arch.children())[0][:]) ## Remove input conv and replace with my own

        self.avg_pool = list(self.old_arch.children())[1]
        self.final_seq = nn.Sequential(*list(self.old_arch.children())[2][:-4],
                                       nn.Linear(in_features=4096, out_features=4096), ## potentially make this map 4096 to 4096 like orig VGG
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(in_features=4096, out_features=256))

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))


    def forward(self,x):


        if self.num_inputs == 1:
            x = x.unsqueeze(1)

        # print("x size: ")
        # print(x.size())

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            # print("curr_x.size(): ")
            # print(curr_x.size())
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            # curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

            out1 = self.first_seq(input)

            # print("out1.size():")
            # print(out1.size())

            out2 = self.avg_pool(out1)

            # print("out2.size():")
            # print(out2.size())

            out2_flat = out2.view(B, -1)

            # print("out2_flat.size():")
            # print(out2_flat.size())

            out3 = self.final_seq(out2_flat)
            x_list.append(out3)

        comb_out = torch.cat(x_list, 1)

        # print("comb_out.size():")
        # print(comb_out.size())

        #comb_out = comb_out.view(B, -1)
        comb_out2 = self.combo_layer(comb_out)
        pred = self.out_layer(comb_out2)

        return pred



class RevisedDenseNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2):
        super(RevisedDenseNet, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.densenet121(pretrained=False)

        self.first_seq = list(self.old_arch.children())[0]

        self.pool = nn.AvgPool2d(kernel_size=(5, 5),padding=(0, 0))

        self.final_lin = nn.Linear(in_features=1024, out_features=256, bias=True)

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))


    def forward(self,x):


        if self.num_inputs == 1:
            x = x.unsqueeze(1)

        # print("x size: ")
        # print(x.size())

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            # print("curr_x.size(): ")
            # print(curr_x.size())
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            # curr_x = curr_x.expand(-1, 3, -1, -1)
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))

            out1 = self.first_seq(input)

            # print("out1.size():")
            # print(out1.size())

            out2 = self.pool(out1)

            # print("out2.size():")
            # print(out2.size())

            out2_flat = out2.view(B, -1)

            # print("out1_flat.size():")
            # print(out2_flat.size())

            out3 = self.final_lin(out2_flat)
            x_list.append(out3)

        comb_out = torch.cat(x_list, 1)

        # print("comb_out.size():")
        # print(comb_out.size())

        #comb_out = comb_out.view(B, -1)
        comb_out2 = self.combo_layer(comb_out)
        pred = self.out_layer(comb_out2)

        return pred
