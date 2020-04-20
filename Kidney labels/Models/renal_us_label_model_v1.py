import torch
from torch import nn
from torch.utils import data
from PIL import Image
import argparse

    ###
    ### DEFINE ULTRASOUND DATASET
    ###

class US_labels_dataset(data.Dataset):
    ## following: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    def __init__(self,ID_list,labels,image_dir):
        self.labels = labels
        self.IDs = ID_list
        self.dir = image_dir

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self,index):
        ID = self.IDs[index]


        X = Image.open(self.dir + ID + '.png') ## ! UPDATE TO CORRECT FOLDER, MAYBE TAKE FOLDER AS ARG?
        y = self.labels[index]

        return X,y

    ###
    ### DEFINE CNN ARCHITECTURE
    ###

class MyCNN(nn.Module):
    def __init__(self, num_conv_filters=18, num_conv_layers = 3, num_labels = 6):
        super(MyCNN, self).__init__()

        self.in_channels = 1
        self.num_conv_layers = num_conv_layers
        self.num_labels = num_labels

        self.l1 = nn.Sequential(nn.Conv2d(self.in_channels, num_conv_filters,5),
                                nn.ReLU(),
                                nn.BatchNorm2d(num_conv_filters),
                                nn.MaxPool2d(2))
        self.lk = nn.Sequential(nn.Conv2d(num_conv_filters, num_conv_filters, 5),
                                nn.ReLU(),
                                nn.BatchNorm2d(num_conv_filters),
                                nn.MaxPool2d(2))

        self.lin1 = nn.Sequential(nn.Linear(in_size, 128),
                                  nn.ReLU())
        self.lin2 = nn.Sequential(nn.Linear(128, self.num_labels))

    def forward(self, x):
        x1 = self.l1(x)

        x_in = x1
        for i in range(self.num_conv_layers):
            x_out = self.lk(x_in)

        x_flat = x_out.view((1,-1))

        x5 = self.lin1(x_flat)
        out = self.lin2(x5)

        return out
        # return x7,x8


    ###
    ### INITIALIZE MODEL
    ###

## Weight initialization code from: Joseph Konan's response on https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
random.seed(1234)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def initialize_training(training_list, data_set_class, network_class,
                        device,
                        val_list=None, is_training_run=True,
                        do_init_weights=True, my_set=analysis_set):

    net = network_class().to(device)
    if do_init_weights:
        net.apply(init_weights)

    if is_training_run:
        training_set = data_set_class(training_list[0], training_list[2], training_list[3], num_chan,
                                      training_list[4], my_set)
        val_set = data_set_class(val_list[0], val_list[2], val_list[3], num_chan, val_list[4], my_set)

        train_loader = DataLoader(training_set, shuffle=True)
        val_loader = DataLoader(val_set, shuffle=False)

        return net, train_loader, val_loader

    else:
        training_set = data_set_class(training_list[0], training_list[2], training_list[3], num_chan,
                                      training_list[4], my_set)
        train_loader = DataLoader(training_set, shuffle=True)

        return net, train_loader

    ###
    ### DEFINE TRAINING LOOP
    ###



    ###
    ### MAIN FUNCTION
    ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--data_dir", default="./", help="Directory of images. Expects 3 folders: "
                                                         "1) 'train', 2) 'val', 3) 'test'"
                                                         "And a .csv file data sheet"
                                                         "With columns 'split','id','label'")
    parser.add_argument("--out_dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    my_nn = MyCNN


if __name__ == "__main__":
    main()
