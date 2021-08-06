import numpy as np
from PIL import Image
import os
import io
import json
import glob
import h5py

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
# QKFIX: See torchmeta.datasets.utils for more informations
from torchmeta.datasets.utils import download_file_generic
from torchmeta.datasets.utils import get_asset


class ilsvrc_2012(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                meta_val=False, meta_test=False, meta_split=None,
                transform=None, target_transform=None, dataset_transform=None,
                class_augmentations=None, download=False):
    dataset = ilsvrc_2012ClassDataset(root, meta_train=meta_train, meta_val=meta_val,
        meta_test=meta_test, meta_split=meta_split, transform=transform,
        class_augmentations=class_augmentations, download=download)
    super().__init__(dataset, num_classes_per_task,
        target_transform=target_transform, dataset_transform=dataset_transform)

class ilsvrc_2012ClassDataset(ClassDataset):
    folder = 'ilsvrc_2012'
     def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super().__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('ilsvrc_2012 integrity check failed')
        self._num_classes = len(self.labels)

        def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return ilsvrc_2012Dataset(index, data, label, transform=transform,
                          target_transform=target_transform)

    # TODO: fix download for ilsvrc2012
    def download(self):
        import tarfile
        import shutil
        import glob
        from tqdm import tqdm

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
            self.tgz_filename, md5=self.tgz_md5)

        tgz_filename = os.path.join(self.root, self.tgz_filename)
        with tarfile.open(tgz_filename, 'r') as f:
            f.extractall(self.root)
        image_folder = os.path.join(self.root, self.image_folder)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label, '*.jpg'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),), dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        tar_folder, _ = os.path.splitext(tgz_filename)
        if os.path.isdir(tar_folder):
            shutil.rmtree(tar_folder)

        attributes_filename = os.path.join(self.root, 'attributes.txt')
        if os.path.isfile(attributes_filename):
            os.remove(attributes_filename)


    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None
