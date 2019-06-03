import pickle
import load_dataset

train_X, train_y, test_X, test_y = load_dataset.load_dataset()
print(type(train_X), type(train_y), type(test_X), type(test_y))