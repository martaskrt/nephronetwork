# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:22:30 2019

@author: larun
"""

import torch 



## OPEN EACH OF THE PTHS -- they're each a dictionary with 1/5 of the data you'll want to use
def open_file(file):
    print("Opening" + file)
    data = torch.load(file)
    #print(data.head)
    #data = pd.read_pickle("preprocessed_images.pickle")
    return data





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder', default=50, type=int, help="Number of epochs")

    args = parser.parse_args()




if __name__ == '__main__':
    main()
