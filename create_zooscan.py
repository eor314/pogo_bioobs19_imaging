import shutil
import os
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_PATH = "/data1/mschroeder/Datasets/19-05-11 ZooScanNet/ZooScanSet/imgs"
DEST_PATH = "/data1/mschroeder/Datasets/19-05-11 MiniZooScanNet"

dataset = ImageFolder(DATASET_PATH)

classes = dataset.classes
dataset = pd.DataFrame(dataset.samples, columns=['path', 'target'])

# Select 30 largest classes
target_counts = dataset["target"].value_counts().iloc[:30]

mask = dataset["target"].isin(target_counts.index)
dataset = dataset[mask]

fraction = 20000.0 / len(dataset)

dset_train, dset_val = train_test_split(
    dataset, stratify=dataset["target"], train_size=fraction, test_size=0.5*fraction)

shutil.rmtree(DEST_PATH, ignore_errors=True)

for phase, dset in (("train", dset_train), ("val", dset_val)):
    for target in target_counts.index:
        class_path = os.path.join(DEST_PATH, phase, classes[target])
        os.makedirs(class_path, exist_ok=True)

    for _, row in dset.iterrows():
        path, target = row
        dest = os.path.join(DEST_PATH, phase, classes[target])
        dest = shutil.copy(path, dest)
        print(path, "->", dest)
