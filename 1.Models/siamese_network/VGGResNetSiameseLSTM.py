import torchvision
import torch
from torch import nn
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MVCNNLstmNet1(nn.Module):
    """
    Multi-View CNN. Pass sag and trans view through independent CNN stacks which outputs
    feature vectors. In MVCNNLstmNet1 we merge the feature vectors by concatenation. This
    is subsequently passed to the LSTM.
    """
    def __init__(self, cnn, shareWeights=False, joinMode=0):
        super(MVCNNLstmNet1, self).__init__()
        # settings
        self.shareWeights = shareWeights
        # joinMode values 0: join stacks by concatting, 1: join stacks with avg pool,
        # 2: join stacks with max pool, 3: join stacks using linear layer
        self.joinMode = joinMode
        # hyper-parameters
        if joinMode == 0:
            self.hidden_dim =  512  # lstm hidden dim
        elif joinMode in (1,2,3):
            self.hidden_dim = 256  # lstm hidden dim

        self.layer_dim = 1  # lstm num of hidden layers
        self.batch_size = 1  # processing 1 kidney sequence at a time
        self.classes = 2  # two classes are surgery required or not required

        # net layers
        self.cnn1 = CNN(cnn)  # used for sag view input
        if not shareWeights:
            self.cnn2 = CNN(cnn)  # used for trans view input

        if joinMode == 3:
            self.comboNet = nn.Sequential(nn.Linear(512, 256, bias=True), nn.ReLU())

        self.lstm = nn.LSTM(input_size=512 if joinMode == 0 else 256,               # dim of each input vector in seq
                            hidden_size=self.hidden_dim,  # dim of hidden layer also output size
                            num_layers=self.layer_dim,    # num of stacked LSTM Cells
                            batch_first=True)

        self.outNet = nn.Linear(self.hidden_dim, self.classes, bias=True)


    def forward(self, x):
        # input of the form [1, m, 2,256,256] 2 images sag and trans, m visits (m = len(x))
        x = torch.squeeze(x, 0)
        if torch.cuda.is_available():
            x = torch.cuda.FloatTensor(x.to(device))
        else:
            x = torch.FloatTensor(x.to(device))

        x_to_lstm = []
        for i in range(len(x)):
            x1, x2 = x[i][0], x[i][1]
            cnn1_out = self.cnn1(x1)  # vector of 256
            if self.shareWeights:
                cnn2_out = self.cnn1(x2)  # vector of 256
            else:
                cnn2_out = self.cnn2(x2)  # vector of 256
            if self.joinMode == 0:
                x_to_lstm.append(torch.cat((torch.squeeze(cnn1_out, 0), torch.squeeze(cnn2_out, 0)), dim=0))
            elif self.joinMode in (1, 2):
                # here we can just take the max or average of the two 256 vectors, not sure if this would do well?
                # alternativly we put a max pool layer before passing sag and trans to cnn stacks and just have
                # 1 cnn stack but this also doesnt seem like a good approach intuitively?
                # although we never know performance until we run it
                pass
            elif self.joinMode == 3:
                x_to_linear_combo = torch.cat((torch.squeeze(cnn1_out, 0), torch.squeeze(cnn2_out, 0)), dim=0)
                x_to_lstm.append(self.comboNet(x_to_linear_combo))

        seq = torch.stack(x_to_lstm).unsqueeze(0) # len 512
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
            densenet = torchvision.models.densenet121(pretrained=False)
            self.seq1 = list(densenet.children())[0]
            self.pool = nn.AvgPool2d(kernel_size=(5, 5), padding=(0,0))
            self.seq2 = nn.Linear(in_features=1024, out_features=256, bias=True)
            self.seq3 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                           nn.ReLU())
        elif "resnet" in cnn:
            if cnn == "resnet18":
                self.seq1 = torchvision.models.resnet18(pretrained=False)
                self.seq1.fc = nn.Linear(512, 256, bias=True)
                self.seq2 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                           nn.ReLU())
            elif cnn == "resnet50":
                self.seq1 = torchvision.models.resnet50(pretrained=False)
                self.seq1.fc = nn.Linear(2048, 256, bias=True)
                self.seq2 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                           nn.ReLU())
        elif "vgg" in cnn:
            if cnn == "vgg":
                vgg = torchvision.models.vgg16(pretrained=False)
            elif cnn == "vgg_bn":
                vgg = torchvision.models.vgg16_bn(pretrained=False)

            self.seq1 = nn.Sequential(*list(vgg.children())[0][:])
            self.pool = list(vgg.children())[1]
            self.seq2 = nn.Sequential(*list(vgg.children())[2][:-4],
                                           nn.Linear(in_features=4096, out_features=4096),
                                           nn.ReLU(),
                                           nn.Dropout(p=0.5),
                                           nn.Linear(in_features=4096, out_features=256))
            self.seq3 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                         nn.ReLU())

    def forward(self, input):
        if self.cnnLabel == "densenet":
            B = 1
            input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # make three channels by replication
            out1 = self.seq1(input)
            out2 = self.pool(out1)
            out2_flat = out2.view(B, -1)
            out3 = self.seq2(out2_flat)
            out4 = self.seq3(out3)
            return out4

        elif "resnet" in self.cnnLabel:
            input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # make three channels by replication
            out1 = self.seq1(input)
            out2 = self.seq2(out1)
            return out2

        elif "vgg" in self.cnnLabel:
            B = 1
            input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # make three channels by replication
            out1 = self.seq1(input)
            out2 = self.pool(out1)
            out2_flat = out2.view(B, -1)
            out3 = self.seq2(out2_flat)
            out4 = self.seq3(out3)
            return out4

        return self.net(input)

class CNN2(nn.Module):
    """
    basic CNN wrapper; takes in one input image with 2 channels 2x256x256 and outputs vector 256
    """
    def __init__(self, cnn="vgg_bn"):
        super(CNN2, self).__init__()
        self.cnnLabel = cnn
        self.cnn3to2 = nn.Conv2d(2,3, kernel_size=7, stride=1, padding=3)
        if cnn == "densenet":
            densenet = torchvision.models.densenet121(pretrained=False)
            # densenet.features[0].in_channels = 2
            # densenet = densenet121(pretrained=False)
            self.seq1 = list(densenet.children())[0]
            self.pool = nn.AvgPool2d(kernel_size=(5, 5), padding=(0,0))
            self.seq2 = nn.Linear(in_features=1024, out_features=256, bias=True)
            self.seq3 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                           nn.ReLU())
        elif "resnet" in cnn:
            if cnn == "resnet18":
                self.seq1 = torchvision.models.resnet18(pretrained=False)
                self.seq1.fc = nn.Linear(512, 256, bias=True)
                self.seq2 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                           nn.ReLU())
            elif cnn == "resnet50":
                self.seq1 = torchvision.models.resnet50(pretrained=False)
                self.seq1.fc = nn.Linear(2048, 256, bias=True)
                self.seq2 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                           nn.ReLU())
        elif "vgg" in cnn:
            if cnn == "vgg":
                vgg = torchvision.models.vgg16(pretrained=False)
            elif cnn == "vgg_bn":
                vgg = torchvision.models.vgg16_bn(pretrained=False)
            self.seq1 = nn.Sequential(*list(vgg.children())[0][:])
            self.pool = list(vgg.children())[1]
            self.seq2 = nn.Sequential(*list(vgg.children())[2][:-4],
                                           nn.Linear(in_features=4096, out_features=4096),
                                           nn.ReLU(),
                                           nn.Dropout(p=0.5),
                                           nn.Linear(in_features=4096, out_features=256))
            self.seq3 = nn.Sequential(nn.Linear(256, 256, bias=True),
                                         nn.ReLU())

    def forward(self, input):
        input = input.unsqueeze(0)
        input = self.cnn3to2(input)
        if self.cnnLabel == "densenet":
            B = 1
            # input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # make three channels by replication
            out1 = self.seq1(input)
            out2 = self.pool(out1)
            out2_flat = out2.view(B, -1)
            out3 = self.seq2(out2_flat)
            out4 = self.seq3(out3)
            return out4

        elif "resnet" in self.cnnLabel:
            # input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # make three channels by replication
            out1 = self.seq1(input)
            out2 = self.seq2(out1)
            return out2

        elif "vgg" in self.cnnLabel:
            B = 1
            # input = input.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # make three channels by replication
            out1 = self.seq1(input)
            out2 = self.pool(out1)
            out2_flat = out2.view(B, -1)
            out3 = self.seq2(out2_flat)
            out4 = self.seq3(out3)
            return out4

        return self.net(input)

class MVCNNLstmNet2(nn.Module):
    """
    MVCNN with 4 stacks of CNN
    Notes: Might be better to make this two stack each stack takes two layer input layers are
    left and right, stack s represent sag and trans
    """
    def __init__(self, cnn):
        pass
    def forward(self, input):
        pass

class BichannelCNNLstmNet(nn.Module):
    """
    1 CNN stack with input being 2 layers(channels) 1 layer each for sag and trans
    """
    def __init__(self, cnn):
        super(BichannelCNNLstmNet, self).__init__()
        # hyper-parameters
        self.hidden_dim = 256  # lstm hidden dim
        self.layer_dim = 1  # lstm num of hidden layers
        self.batch_size = 1  # processing 1 kidney sequence at a time
        self.classes = 2  # two classes are surgery required or not required

        # net layers
        self.cnn1 = CNN2(cnn)  # used for sag view input

        self.lstm = nn.LSTM(input_size=256,               # dim of each input vector in seq
                            hidden_size=self.hidden_dim,  # dim of hidden layer also output size
                            num_layers=self.layer_dim,    # num of stacked LSTM Cells
                            batch_first=True)

        self.outNet = nn.Sequential(nn.Linear(self.hidden_dim, self.classes, bias=True),
                                    nn.ReLU())
    def forward(self, input):
        x = torch.squeeze(input, 0)
        if torch.cuda.is_available():
            x = torch.cuda.FloatTensor(x.to(device))
        else:
            x = torch.FloatTensor(x.to(device))

        x_to_lstm = []
        for i in range(len(x)):
            # x1, x2 = x[i][0], x[i][1]
            cnn1_out = self.cnn1(x[i])  # vector of 256
            x_to_lstm.append(torch.squeeze(cnn1_out, 0))

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


class QuadchannelCNNLstmNet(nn.Module):
    """
    1 CNN stack with input being 4 layers 1 for each left_sag, left_trans, right_sag, right_trans
    Notes: this is going to require different type of input as its on a per patient basis not per kidney

    """
    def __init__(self, cnn):
        pass
    def forward(self, input):
        pass

class CNN3(nn.Module):
    def __init__(self, cnn):
        super(CNN3, self).__init__()
        self.cnnLabel = cnn
        if cnn == "densenet":
            self.model = RevisedDenseNet()
        elif "resnet" in cnn:
            if cnn == "resnet18":
                self.model = RevisedResNet()
            elif cnn == "resnet50":
                pass
        elif "vgg" in cnn:
            if cnn == "vgg":
                pass
            elif cnn == "vgg_bn":
                self.model = RevisedVGG_bn()
    def forward(self, input):
        cnn =self.cnnLabel
        return self.model(input)
        # if cnn == "densenet":
        #     pass
        # elif "resnet" in cnn:
        #     if cnn == "resnet18":
        #         pass
        #     elif cnn == "resnet50":
        #         pass
        # elif "vgg" in cnn:
        #     if cnn == "vgg":
        #         pass
        #     elif cnn == "vgg_bn":
        #         pass

class CNNLstm_Wrapper(nn.Module):
    def __init__(self, cnn):
        super(CNNLstm_Wrapper, self).__init__()
        self.cnnLabel = cnn
        self.classes = 2
        self.num_inputs = 2  # not used kept here for mention
        # lstm hyperparams
        self.batch_size = 1
        self.layer_dim = 1
        self.hidden_dim = 256

        self.cnn = CNN3(cnn)
        self.lstm = nn.LSTM(input_size=256,               # dim of each input vector in seq
                            hidden_size=self.hidden_dim,  # hyperparam
                            num_layers=self.layer_dim,    # num of stacked LSTM Cells
                            batch_first=True)             # put batch_size as first dim of input seq
        self.out_layer = nn.Sequential(nn.Linear(256, self.classes, bias=True), nn.ReLU())

    def forward(self, kidney):
        # kidney is a seq of images for one kidney
        # process all images through CNN then pass through lstm
        x_to_lstm = []
        # create lstm sequence by running images for kidney through CNN
        for j in range(len(kidney)):
            x = kidney[j]
            cnn_out = self.cnn(x)  # 256 vector
            x_to_lstm.append(cnn_out)

        seq = torch.stack(x_to_lstm)
        # init hidden and cell state for this seq of images
        hidden_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        cellState_0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_()
        # process through lstm in a 1 batch of whole sequence
        lstm_out,_ = self.lstm(seq, (hidden_0.detach(), cellState_0.detach()))
        # squeeze out batch size
        lstm_out = torch.squeeze(lstm_out, 0)
        pred = self.out_layer(lstm_out)
        # return a prediction that looks like e.g. [p1,p2,p3] where p is prob for each visit
        return pred

class RevisedResNet(nn.Module):
    def __init__(self):
        super(RevisedResNet, self).__init__()
        self.old_arch = torchvision.models.resnet18(pretrained=False)
        self.old_arch.fc = nn.Linear(512, 256, bias=True)
        self.combo_layer = nn.Sequential(nn.Linear(512, 256, bias=True), nn.ReLU())
    def forward(self, x):
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            input = torch.cuda.FloatTensor(curr_x.to(device)) if torch.cuda.is_available() else torch.FloatTensor(curr_x.to(device))
            res_out = self.old_arch(input)
            x_list.append(res_out)
        comb_out = torch.cat(x_list, 1)
        comb_out = comb_out.view(B, -1)
        comb_out2 = self.combo_layer(comb_out)
        return comb_out2

class RevisedVGG_bn(nn.Module):
    def __init__(self):
        super(RevisedVGG_bn, self).__init__()
        self.old_arch = torchvision.models.vgg16_bn(pretrained=False)
        self.first_seq = nn.Sequential(*list(self.old_arch.children())[0][:]) ## Remove input conv and replace with my own
        self.avg_pool = list(self.old_arch.children())[1]
        self.final_seq = nn.Sequential(*list(self.old_arch.children())[2][:-4],
                                       nn.Linear(in_features=4096, out_features=4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(in_features=4096, out_features=256))
        self.combo_layer = nn.Sequential(nn.Linear(512, 256, bias=True),
                                         nn.ReLU())
    def forward(self,x):
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            input = torch.cuda.FloatTensor(curr_x.to(device)) if torch.cuda.is_available() else torch.FloatTensor(curr_x.to(device))
            out1 = self.first_seq(input)
            out2 = self.avg_pool(out1)
            out2_flat = out2.view(B, -1)
            out3 = self.final_seq(out2_flat)
            x_list.append(out3)
        comb_out = torch.cat(x_list, 1)
        comb_out2 = self.combo_layer(comb_out)
        return comb_out2

class RevisedDenseNet(nn.Module):
    def __init__(self):
        super(RevisedDenseNet, self).__init__()
        self.old_arch = torchvision.models.densenet121(pretrained=False)
        self.first_seq = list(self.old_arch.children())[0]
        self.pool = nn.AvgPool2d(kernel_size=(5, 5),padding=(0, 0))
        self.final_lin = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.combo_layer = nn.Sequential(nn.Linear(512, 256, bias=True),nn.ReLU())
    def forward(self,x):
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            input = torch.cuda.FloatTensor(curr_x.to(device)) if torch.cuda.is_available() else torch.FloatTensor(curr_x.to(device))
            out1 = self.first_seq(input)
            out2 = self.pool(out1)
            out2_flat = out2.view(B, -1)
            out3 = self.final_lin(out2_flat)
            x_list.append(out3)
        comb_out = torch.cat(x_list, 1)
        comb_out2 = self.combo_layer(comb_out)
        return comb_out2

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

'''
usefule code to visualize image
import matplotlib.pyplot as plt
imgplot = plt.imshow(img.permute(1, 2, 0))
'''