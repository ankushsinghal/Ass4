from __future__ import print_function
import pandas as pd
import numpy as np
import sys

#Read training data
train_dataset = pd.read_csv(sys.argv[1]).as_matrix()
train_labels = train_dataset[:, 0]
train_dataset = np.delete(train_dataset, 0, 1)
print("Training data", train_dataset.shape)

#Read test data
test_dataset = pd.read_csv(sys.argv[2]).as_matrix()
print("Test data", test_dataset.shape)

predictions = np.zeros(test_dataset.shape[0], dtype=np.int64)

dataFrame = pd.DataFrame(predictions, columns=["Label"])
dataFrame.index += 1
dataFrame.to_csv("output.csv", index_label="ImageId")
print("Generated output.csv")