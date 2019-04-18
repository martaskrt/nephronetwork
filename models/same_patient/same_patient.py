import argparse
import itertools
import importlib.machinery
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Import custom helper modules
load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../preprocess/load_dataset.py').load_module()

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = y

    def __getitem__(self, index):
        imgs, target = self.X[index], self.y[index]
        return imgs, target

    def __len__(self):
        return len(self.X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=70, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--datafile", default="../../data/preprocessed_images_20190315_25%.pkl", help="File containing pandas dataframe with images stored as numpy array")

    args = parser.parse_args()

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    train_X, _, train_features, test_X, _, test_features = load_dataset.load_dataset(views_to_get="sag", get_features=True, pickle_file=args.datafile)
    train_ids = train_features['study_id']
    test_ids = test_features['study_id']

    # Validate
    if len(train_X) != len(train_ids) or len(test_X) != len(test_ids):
        raise Exception('Length mismatch')

    # Create pairs
    training_pairs = [] # [[image_1, study_id_1], [image_2, study_id_2], ... ]
    test_pairs = []

    for i in range(len(train_X)):
        training_pairs.append([train_X[i], train_ids[i]])

    for i in range(len(test_X)):
        test_pairs.append([test_X[i], test_ids[i]])

    train_X = test_X = []

    # Create combinations
    # [([image_1, study_id_1], [image_2, study_id_2]), ... ]
    training_combinations = list(itertools.combinations(training_pairs, 2))
    for i, combo in enumerate(training_combinations):
        training_combinations[i] = [combo[0][0], combo[0][1], combo[1][0], combo[1][1]]
    test_combinations = list(itertools.combinations(test_pairs, 2))
    for i, combo in enumerate(test_combinations):
        test_combinations[i] = [combo[0][0], combo[0][1], combo[1][0], combo[1][1]]
    # [[image_1, study_id_1, image_2, study_id_2], ... ]

    training_pairs = []  # [[image_1, study_id_1], [image_2, study_id_2], ... ]
    test_pairs = []

    training_combinations_y = np.zeros(len(training_combinations), dtype=np.int8)
    for i, combo in enumerate(training_combinations):
        if combo[1] == combo[3]:
            training_combinations_y[i] = 1
        else:
            training_combinations_y[i] = 0
        training_combinations[i] = [combo[0], combo[2]]

    test_combinations_y = np.zeros(len(test_combinations),dtype=np.int8)
    for i, combo in enumerate(test_combinations):
        if combo[1] == combo[3]:
            test_combinations_y[i] = 1
        else:
            test_combinations_y[i] = 0
        test_combinations[i] = [combo[0], combo[2]]

    limit = 50
    training_combinations = np.asarray(training_combinations[:limit])
    training_combinations_y = training_combinations_y[:5000]
    test_combinations = np.asarray(test_combinations[:limit])
    test_combinations_y = test_combinations_y[:limit]


    # train_X = [#batch, 2, 256, 256]
    training_set = KidneyDataset(training_combinations, training_combinations_y)
    training_generator = DataLoader(training_set, **params)

    validation_set = KidneyDataset(test_combinations, test_combinations_y)
    validation_generator = DataLoader(validation_set, **params)


if __name__ == '__main__':
    main()