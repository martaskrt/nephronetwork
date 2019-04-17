import argparse
import itertools
import importlib.machinery


# Import custom helper modules
load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../preprocess/load_dataset.py').load_module()

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


    # TODO: use load_dataset.py
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

    # Create combinations
    # [[image_1, study_id_1, image_2, study_id_2], ... ]
    training_combinations = list(itertools.combinations(training_pairs, 2))
    test_combinations = list(itertools.combinations(test_pairs, 2))

    for i, combo in enumerate(training_combinations):
        if combo[0][1] == combo[1][1]:
            training_combinations[i] = [combo[0][0], combo[1][0], 1]
        else:
            training_combinations[i] = [combo[0][0], combo[1][0], 0]
            
    for i, combo in enumerate(test_combinations):
        if combo[0][1] == combo[1][1]:
            test_combinations[i] = [combo[0][0], combo[1][0], 1]
        else:
            test_combinations[i] = [combo[0][0], combo[1][0], 0]


if __name__ == '__main__':
    main()