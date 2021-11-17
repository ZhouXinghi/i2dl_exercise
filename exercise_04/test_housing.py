import sys
sys.path.append("$HOME/Documents/i2dl_exercise_debug/exercise_04")
from exercise_code.data.csv_dataset import CSVDataset
from exercise_code.data.csv_dataset import FeatureSelectorAndNormalizationTransform
from exercise_code.data.dataloader import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
root_path = os.path.join(i2dl_exercises_path, "datasets", 'housing')
housing_file_path = os.path.join(root_path, "housing_train.csv")
download_url = 'https://cdn3.vision.in.tum.de/~dl4cv/housing_train.zip'

# Always make sure this line was run at least once before trying to
# access the data manually, as the data is downloaded in the 
# constructor of CSVDataset.
target_column = 'SalePrice'
train_dataset = CSVDataset(target_column=target_column, root=root_path, download_url=download_url, mode="train")

train_dataset.df.corr()[target_column].sort_values(ascending=False)[:3]


# selected feature and target 
train_dataset.df[['GrLivArea',target_column]]

plt.scatter(train_dataset.df[['GrLivArea']], train_dataset.df[[target_column]])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")


df = train_dataset.df
# Select a feature to keep plus the target column.
selected_columns = ['GrLivArea', target_column]
mn, mx, mean = df.min(), df.max(), df.mean()

column_stats = {}
for column in selected_columns:
    crt_col_stats = {'min' : mn[column],
                     'max' : mx[column],
                     'mean': mean[column]}
    column_stats[column] = crt_col_stats    

transform = FeatureSelectorAndNormalizationTransform(column_stats, target_column)

def rescale(data, key = "SalePrice", column_stats = column_stats):
    """ Rescales input series y"""
    mx = column_stats[key]["max"]
    mn = column_stats[key]["min"]

    return data * (mx - mn) + mn



# Always make sure this line was run at least once before trying to
# access the data manually, as the data is downloaded in the 
# constructor of CSVDataset.
train_dataset = CSVDataset(mode="train", target_column=target_column, root=root_path, download_url=download_url, transform=transform)
val_dataset = CSVDataset(mode="val", target_column=target_column, root=root_path, download_url=download_url, transform=transform)
test_dataset = CSVDataset(mode="test", target_column=target_column, root=root_path, download_url=download_url, transform=transform)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))
print("Number of test samples:", len(test_dataset))




# load training data into a matrix of shape (N, D), same for targets resulting in the shape (N, 1)
X_train = [train_dataset[i]['features'] for i in range((len(train_dataset)))]
X_train = np.stack(X_train, axis=0)
y_train = [train_dataset[i]['target'] for i in range((len(train_dataset)))]
y_train = np.stack(y_train, axis=0)
print("train data shape:", X_train.shape)
print("train targets shape:", y_train.shape)

# load validation data
X_val = [val_dataset[i]['features'] for i in range((len(val_dataset)))]
X_val = np.stack(X_val, axis=0)
y_val = [val_dataset[i]['target'] for i in range((len(val_dataset)))]
y_val = np.stack(y_val, axis=0)
print("val data shape:", X_val.shape)
print("val targets shape:", y_val.shape)

# load test data
X_test = [test_dataset[i]['features'] for i in range((len(test_dataset)))]
X_test = np.stack(X_test, axis=0)
y_test = [test_dataset[i]['target'] for i in range((len(test_dataset)))]
y_test = np.stack(y_test, axis=0)
print("test data shape:", X_test.shape)
print("test targets shape:", y_test.shape)


# 0 encodes small prices, 1 encodes large prices.





from exercise_code.networks.utils import binarize
y_all = np.concatenate([y_train, y_val, y_test])
thirty_percentile = np.percentile(y_all, 30)
seventy_percentile = np.percentile(y_all, 70)

# Prepare the labels for classification.
X_train, y_train = binarize(X_train, y_train, thirty_percentile, seventy_percentile )
X_val, y_val   = binarize(X_val, y_val, thirty_percentile, seventy_percentile)
X_test, y_test  = binarize(X_test, y_test, thirty_percentile, seventy_percentile)

print("train data shape:", X_train.shape)
print("train targets shape:", y_train.shape)
print("val data shape:", X_val.shape)
print("val targets shape:", y_val.shape)
print("test data shape:", X_test.shape)
print("test targets shape:", y_test.shape)
