import torch
from torch.utils.data import Dataset

from torchmetal.datasets.helpers import (omniglot, quickdraw, aircraft,
                                         cubirds, dtd, vggflower,
                                         trafficsigns, fungi, mscoco,
                                         ilsvrc2012)


class MetaDataset(Dataset):
    """
    A dataset of meta datasets. This acts as a list for choosing, at random,
    an episode comprised of one of these meta datasets, for each episode. All
    transforms will act on all datasets for batching purposes.

    Parameters
    ----------
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmetal.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmetal.transforms.HorizontalFlip()`.
    """

    def __init__(self,
                 meta_train=False,
                 meta_test=False,
                 meta_val=False,
                 download=True,
                 transform=None,
                 target_transform=None,
                 dataset_transform=None,
                 class_augmentation=None,
                 ):
        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_val = meta_val
        self.meta_dataset = True

        self.datasets = [
            omniglot(
                "data",
                use_vinyals_split=False,
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            quickdraw(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            aircraft(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            cubirds(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            dtd(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            vggflower(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            trafficsigns(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            fungi(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            mscoco(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
            ilsvrc2012(
                "data",
                meta_dataset=True,
                meta_train=True,
                download=True,
            ),
        ]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        if isinstance(index, tuple):  # If index also has sampling data return get item of dataset
            return (self.datasets[index[1]][index[0]])
        else:  # If index is only asking for dataset return actual dataset
            return (self.datasets[index])
