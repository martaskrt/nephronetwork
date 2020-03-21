import torchvision
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MVCNNLstmNet1(nn.Module):
    """
    Multi-View CNN. Pass sag and trans view through independent CNN stacks which outputs
    feature vectors. In MVCNNLstmNet1 we merge the feature vectors by concatenation. This
    is subsequently passed to the LSTM.
    """
    def __init__(self, cnn):
        super(MVCNNLstmNet1, self).__init__()
        # hyper-parameters
        self.hidden_dim =  512 # lstm hidden dim
        self.layer_dim = 1  # lstm num of hidden layers
        self.batch_size = 1 # processing 1 kidney sequence at a time
        self.classes = 2 # two classes are surgery required or not required

        # net layers
        self.cnn1 = CNN(cnn)  # used for sag view input
        self.cnn2 = CNN(cnn)  # used for trans view input

        self.lstm = nn.LSTM(input_size=512,               # dim of each input vector in seq
                            hidden_size=self.hidden_dim,  # dim of hidden layer also output size
                            num_layers=self.layer_dim,    # num of stacked LSTM Cells
                            batch_first=True)

        self.outNet = nn.Linear(self.hidden_dim, self.classes, bias=True)


    def forward(self, x):
        # input of the form [1, m, 2,256,256] 2 images sag and trans, m visits (m = len(x))
        x = torch.squeeze(x, 0)
        x_to_lstm = []
        for i in range(len(x)):
            x1, x2 = x[i][0], x[i][1]
            cnn1_out = self.cnn1(x1)  # vector of 256
            cnn2_out = self.cnn2(x2)  # vector of 256
            x_to_lstm.append(torch.cat((torch.squeeze(cnn1_out, 0), torch.squeeze(cnn2_out, 0)), dim=0))

        seq = torch.stack(x_to_lstm).unsqueeze(0)
        # init hidden and cell state for this seq of images
        hidden_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        cellState_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        # process whole sequence through lstm
        lstm_out, _ = self.lstm(seq, (hidden_0.detach(), cellState_0.detach()))

        # squeeze out batch size
        lstm_out = torch.squeeze(lstm_out, 0)

        # pass tensor of shape [m,self.hidden_dim] to linear layer which should process
        # each tensor individually with batch mode
        pred = self.outNet(lstm_out)
        # return a prediction that looks like e.g. [0,0,1,0]; len(pred) = m
        return pred


class CNN(nn.Module):
    """
    basic CNN wrapper; takes in one input image 256x256 and outputs vector 256
    """
    def __init__(self, cnn="vgg_bn"):
        super(CNN, self).__init__()
        self.cnnLabel = cnn
        if cnn == "densenet":
            pass
        elif "restnet" in cnn:
            if cnn == "resnet18":
                pass
            elif cnn == "resnet50":
                pass
        elif "vgg" in cnn:
            if cnn == "vgg":
                vgg = torchvision.models.vgg16(pretrained=False)
            elif cnn == "vgg_bn":
                vgg = torchvision.models.vgg16_bn(pretrained=False)

            self.first_seq = nn.Sequential(*list(vgg.children())[0][:])
            self.avg_pool = list(vgg.children())[1]
            self.first_seq = nn.Sequential(self.first_seq, self.avg_pool)
            self.final_seq = nn.Sequential(*list(vgg.children())[2][:-4],
                                           nn.Linear(in_features=4096, out_features=4096),
                                           nn.ReLU(),
                                           nn.Dropout(p=0.5),
                                           nn.Linear(in_features=4096, out_features=256))

    def forward(self, input):
        if "vgg" in self.cnnLabel:
            B = 1
            input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
            out1 = self.first_seq(input)
            out1_flat = out1.view(B, -1)
            return self.final_seq(out1_flat)

        return self.net(input)



class RevisedResNetLstm(nn.Module):
    def __init__(self, pretrain=False, classes=2, num_inputs=2):
        super(RevisedResNetLstm, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        # lstm hyperparams
        self.batch_size = 1
        self.layer_dim = 1
        self.hidden_dim = 256

        self.old_arch = torchvision.models.resnet18(pretrained=pretrain)
        # self.old_arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.old_arch.fc = nn.Linear(512, 256, bias=True)

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.lstm = nn.LSTM(input_size=256,               # dim of each input vector in seq
                            hidden_size=self.hidden_dim,  # hyperparam
                            num_layers=self.layer_dim,    # num of stacked LSTM Cells
                            batch_first=True)             # put batch_size as first dim of input seq
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))

    def forward(self, kidney):
        # kidney is a seq of images for one kidney
        # process all images through CNN then pass through lstm
        x_to_lstm = []
        # create lstm sequence by running images for kidney through CNN
        for j in range(len(kidney)):
            x = kidney[j]
            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            # print("x size: ")
            # print(x.size())

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                # if self.num_inputs == 1:
                #     curr_x = torch.unsqueeze(x[i], 1)
                # else:
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
            x_to_lstm.append(comb_out2)


        seq = torch.stack(x_to_lstm)
        # init hidden and cell state for this seq of images
        hidden_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        cellState_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        # process through lstm in a 1 batch of whole sequence
        lstm_out, hidden = self.lstm( seq, (hidden_0.detach(), cellState_0.detach()) )

        #squeeze out batch size
        lstm_out = torch.squeeze(lstm_out, 0)
        pred = self.out_layer(lstm_out)
        # return a prediction that looks like e.g. [0,0,1,0]
        return pred

class RevisedResNet50(nn.Module):
    def __init__(self, pretrain=False, classes=2, num_inputs=2):
        super(RevisedResNet50, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        # lstm hyperparams
        self.batch_size = 1
        self.layer_dim = 1
        self.hidden_dim = 256

        self.old_arch = torchvision.models.resnet50(pretrained=pretrain)
        #self.old_arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.old_arch.fc = nn.Linear(2048, 256, bias=True)

        self.combo_layer = nn.Sequential(nn.Linear(self.num_inputs * 256, 256, bias=True),
                                         nn.ReLU())
        self.lstm = nn.LSTM(input_size=256,               # dim of each input vector in seq
                            hidden_size=self.hidden_dim,  # hyperparam
                            num_layers=self.layer_dim,    # num of stacked LSTM Cells
                            batch_first=True)             # put batch_size as first dim of input seq
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True))

    def forward(self, x):
        # kidney is a seq of images for one kidney
        # process all images through CNN then pass through lstm
        x_to_lstm = []
        # create lstm sequence by running images for kidney through CNN
        for j in range(len(kidney)):
            x = kidney[j]
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

            comb_out = torch.cat(x_list, 1)
            comb_out = comb_out.view(B, -1)
            comb_out2 = self.combo_layer(comb_out)
            x_to_lstm.append(comb_out2)

        seq = torch.stack(x_to_lstm)
        # init hidden and cell state for this seq of images
        hidden_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        cellState_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        # process through lstm in a 1 batch of whole sequence
        lstm_out, hidden = self.lstm(seq, (hidden_0.detach(), cellState_0.detach()))

        # squeeze out batch size should iterate on for loop
        lstm_out = torch.squeeze(lstm_out, 0)
        pred = self.out_layer(lstm_out)
        return pred

class RevisedVGG(nn.Module):
    def __init__(self, pretrain=False, classes=2, num_inputs=2):
        super(RevisedVGG, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.vgg16(pretrained=pretrain)

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
    def __init__(self, pretrain=False, classes=2, num_inputs=2):
        super(RevisedVGG_bn, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.vgg16_bn(pretrained=pretrain)

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
    def __init__(self, pretrain=False, classes=2, num_inputs=2):
        super(RevisedDenseNet, self).__init__()

        self.classes = classes
        self.num_inputs = num_inputs
        self.old_arch = torchvision.models.densenet121(pretrained=pretrain)

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
