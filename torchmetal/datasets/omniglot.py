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


class Omniglot(CombinationMetaDataset):
    """
    The Omniglot dataset [1]. A dataset of 1623 handwritten characters from
    50 different alphabets.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `omniglot` exists.

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

    use_vinyals_split : bool (default: `True`)
        If set to `True`, the dataset uses the splits defined in [3]. If `False`,
        then the meta-train split corresponds to `images_background`, and the
        meta-test split corresponds to `images_evaluation` (raises an error when
        calling the meta-validation split).

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

    download : bool (default: `False`)
        If `True`, downloads the zip files and processes the dataset in the root
        directory (under the `omniglot` folder). If the dataset is already
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from the original [Omniglot repository]
    (https://github.com/brendenlake/omniglot). The meta train/validation/test
    splits used in [3] are taken from [this repository]
    (https://github.com/jakesnell/prototypical-networks). These splits are
    over 1028/172/423 classes (characters).

    References
    ----------
    .. [1] Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level
           concept learning through probabilistic program induction. Science, 350(6266),
           1332-1338 (http://www.sciencemag.org/content/350/6266/1332.short)

    .. [2] Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2019). The Omniglot
           Challenge: A 3-Year Progress Report (https://arxiv.org/abs/1902.03477)

    .. [3] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
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
        use_vinyals_split=True,
        transform=None,
        target_transform=None,
        dataset_transform=None,
        class_augmentations=None,
        download=False,
    ):
        dataset = OmniglotClassDataset(
            root,
            meta_dataset=meta_dataset,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            use_vinyals_split=use_vinyals_split,
            transform=transform,
            meta_split=meta_split,
            class_augmentations=class_augmentations,
            download=download,
        )
        super(Omniglot, self).__init__(
            dataset,
            num_classes_per_task,
            target_transform=target_transform,
            dataset_transform=dataset_transform,
        )


class OmniglotClassDataset(ClassDataset):
    folder = "omniglot"
    download_url_prefix = "https://github.com/brendenlake/omniglot/raw/master/python"
    zips_md5 = {
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
    }


    def __init__(
        self,
        root,
        meta_dataset=False,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        use_vinyals_split=True,
        transform=None,
        class_augmentations=None,
        download=False,
    ):
        super(OmniglotClassDataset, self).__init__(
            meta_dataset=meta_dataset,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=class_augmentations,
        )

        if meta_dataset and meta_train or meta_test or meta_val:
            raise ValueError(
                "Trying to use a meta_split with the "
                "meta_dataset sampler. You must set all meta_splits to "
                " False ")
        if self.meta_val and (not use_vinyals_split):
            raise ValueError(
                "Trying to use the meta-validation without the "
                "Vinyals split. You must set `use_vinyals_split=True` to use "
                "the meta-validation split."
            )

        self.root = Path(root).resolve() / 'omniglot'
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform

        self.data_filename = self.root / 'records/data.h5'
        self.dataset_spec = self.root / 'records/dataset_spec.json'

        # TODO: logic to decide on how to load chosen split labels
        self.split_labels = self.root / 'splits/omniglot_splits.json'

        self._data = None
        self._labels = None
        self._splits = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Omniglot integrity check failed")
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        if not self.meta_dataset:
            character_name = self.labels[index % self.num_classes][1]
            data = self.data[self.labels[index % self.num_classes][0]]
            transform = self.get_transform(index, self.transform)
            target_transform = self.get_target_transform(index)

            return OmniglotDataset(
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

            return OmniglotDataset(
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
            with open(self.split_labels, "r") as f:
                self._labels = json.load(f)
                if self.use_vinyals_split:
                    if self.meta_train:
                        self._labels = self._labels['vinyals_train']
                    elif self.meta_test:
                        self._labels = self._labels['vinyals_test']
                    elif self.meta_val:
                        self._labels = self._labels['vinyals_val']
                elif self.meta_dataset:
                    with open(self.dataset_spec, 'r') as f:
                        self._labels = json.load(f)
                    self._labels = self._labels['class_names']
                else:
                    if self.meta_train:
                        self._labels = self._labels['train']
                    elif self.meta_test:
                        self._labels = self._labels['test']
        return self._labels

    @property
    def splits(self):
        if self._splits is None:
            with open(self.split_labels, 'r') as f:
                self._splits = json.load(f)
        return self._splits

    def _check_integrity(self):
        return(
            self.data_filename.is_file() and
            self.dataset_spec.is_file() and
            self.split_labels.is_file())

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import zipfile

        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = "{0}.zip".format(name)
            filename = self.root / zip_filename
            if filename.is_file():
                continue

            url = "{0}/{1}".format(self.download_url_prefix, zip_filename)
            download_url(url, str(self.root), zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, "r") as f:
                f.extractall(self.root / "data")

        convert(self.root, dataset="omniglot")


class OmniglotDataset(Dataset):
    def __init__(
        self, index, data, character_name, transform=None, target_transform=None
    ):
        super(OmniglotDataset, self).__init__(
            index, transform=transform, target_transform=target_transform
        )
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index]))
        image = image.convert("L")
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
