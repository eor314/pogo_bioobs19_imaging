from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.dataset import Subset


def stratified_random_split(dataset, **kwargs):
    """
    Parameters:
        dataset: PyTorch Dataset
        **kwargs: see train_test_split
            test_size, train_size, random_state, shuffle
    """
    indices = np.arange(len(dataset))

    stratify = kwargs.pop("stratify", dataset.targets)

    indices_train, indices_test = train_test_split(
        indices, stratify=stratify, **kwargs)

    return (Subset(dataset, indices_train),
            Subset(dataset, indices_test))
