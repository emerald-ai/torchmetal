import glob
import io
import json
import os
from pathlib import Path

import h5py
from PIL import Image, ImageOps
from torchvision.datasets.utils import download_url, list_dir

from torchmetal.datasets.utils import get_asset
from torchmetal.utils.data import ClassDataset, CombinationMetaDataset, Dataset
from torchmetal.datasets.metadataset.dataset_conversion import convert


class Fungi(CombinationMetaDataset):
    """
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `fungi` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.

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

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.

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

    convert : bool (default: `False`)
        If `True`, converts the npy files and processes the dataset in the root
        directory (under the `fungi` folder). If the dataset is already
        available, this does not convert the dataset again.
    """

    def __init__(
        self,
        root,
        num_classes_per_task=None,
        meta_dataset=False,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        transform=None,
        target_transform=None,
        dataset_transform=None,
        class_augmentations=None,
        download=False,
    ):
        dataset = FungiClassDataset(
            root,
            meta_dataset=meta_dataset,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            transform=transform,
            meta_split=meta_split,
            class_augmentations=class_augmentations,
            download=download,
        )
        super(Fungi, self).__init__(
            dataset,
            num_classes_per_task,
            target_transform=target_transform,
            dataset_transform=dataset_transform,
        )


class FungiClassDataset(ClassDataset):
    def __init__(
        self,
        root,
        meta_dataset=False,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        transform=None,
        class_augmentations=None,
        download=False,
    ):
        super(FungiClassDataset, self).__init__(
            meta_dataset=meta_dataset,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=class_augmentations,
        )

        if meta_val and (not meta_dataset):
            raise ValueError(
                "Trying to use the meta-validation without the "
                "Vinyals split. You must set `use_vinyals_split=True` to use "
                "the meta-validation split."
            )
        if meta_dataset and (not meta_train and not meta_test and
                             not meta_val):
            raise ValueError(
                """
                Trying to use meta_dataset without split for sampling.
                Must use meta_train, meta_test, or meta_val.
                """
            )

        self.root = Path(root).resolve() / 'fungi'
        self.transform = transform
        self.data_filename = self.root / 'records/data.h5'
        self.dataset_spec = self.root / 'records/dataset_spec.json'
        self.split_labels = self.root / 'splits/fungi_splits.json'

        self._data = None
        self._labels = None
        self._splits = None

        if download:
            self.convert()

        if not self._check_integrity():
            raise RuntimeError("fungi integrity check failed")
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        if not self.meta_dataset:
            character_name = self.labels[index % self.num_classes][1]
            data = self.data[self.labels[index % self.num_classes][0]]
            transform = self.get_transform(index, self.transform)
            target_transform = self.get_target_transform(index)

            return FungiDataset(
                index,
                data,
                character_name,
                transform=transform,
                target_transform=target_transform,
            )

        else:
            character_name = self.labels[str(index)]
            data = self.data[str(index)]
            transform = self.get_transform(index, self.transform)
            target_transform = self.get_target_transform(index)

            return FungiDataset(
                index,
                data,
                character_name,
                transform=transform,
                target_transform=target_transform,
            )

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.data_filename, "r")
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.dataset_spec, 'r') as f:
                self._labels = json.load(f)
            self._labels = self._labels['class_names']
        return self._labels

    def _check_integrity(self):
        return(
            self.data_filename.is_file() and
            self.dataset_spec.is_file() and
            self.split_labels.is_file())

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def convert(self):
        if self._check_integrity():
            return

        convert(self.root, dataset="fungi")


class FungiDataset(Dataset):
    def __init__(
        self, index, data, character_name, transform=None, target_transform=None
    ):
        super(FungiDataset, self).__init__(
            index, transform=transform, target_transform=target_transform
        )
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index]))
        image = image.convert('L')
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
