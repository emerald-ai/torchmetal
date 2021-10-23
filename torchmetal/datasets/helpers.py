import warnings

from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from torchmetal.datasets import (CIFARFS, CUB, FC100, DoubleMNIST,
                                 MiniImagenet, Omniglot, Pascal5i,
                                 TieredImagenet, TripleMNIST, Quickdraw,
                                 Aircraft, CuBirds, Dtd, VggFlower,
                                 TrafficSigns, Mscoco, Fungi, Ilsvrc2012)
from torchmetal.transforms import (Categorical, ClassSplitter, Rotation,
                                   SegmentationPairTransform)

__all__ = [
    "omniglot",
    "quickdraw",
    "aircraft",
    "cubirds",
    "miniimagenet",
    "tieredimagenet",
    "cifar_fs",
    "fc100",
    "cub",
    "doublemnist",
    "triplemnist",
    "dtd",
    "vggflower",
    "trafficsigns",
    "mscoco",
    "fungi",
    "ilsvrc2012",
]


def helper_with_default(
    klass,
    folder,
    shots=None,
    ways=None,
    meta_dataset=None,
    shuffle=True,
    test_shots=None,
    seed=None,
    defaults={},
    **kwargs
):
    if "num_classes_per_task" in kwargs:
        warnings.warn(
            "Both arguments `ways` and `num_classes_per_task` were "
            "set in the helper function for the number of classes per task. "
            "Ignoring the argument `ways`.",
            stacklevel=2,
        )
        ways = kwargs["num_classes_per_task"]
    if "transform" not in kwargs:
        kwargs["transform"] = defaults.get("transform", ToTensor())
    if "target_transform" not in kwargs and not meta_dataset:
        kwargs["target_transform"] = defaults.get("target_transform", Categorical(ways))
    elif "target_transform" not in kwargs and meta_dataset:
        kwargs["target_transform"] = defaults.get("target_transform", Categorical())
    if "class_augmentations" not in kwargs:
        kwargs["class_augmentations"] = defaults.get("class_augmentations", None)
    if test_shots is None:
        test_shots = shots

    if not meta_dataset:
        dataset = klass(folder, num_classes_per_task=ways, **kwargs)
        dataset = ClassSplitter(
            dataset,
            shuffle=shuffle,
            num_train_per_class=shots,
            num_test_per_class=test_shots,
        )
        dataset.seed(seed)
    else:
        dataset = klass(folder, meta_dataset=meta_dataset, **kwargs)
        dataset = ClassSplitter(
            dataset,
            shuffle=shuffle,
            variable_class_split=True,
        )
        dataset.seed(seed)

    return dataset


def omniglot(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize(28), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        Omniglot,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def quickdraw(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        Quickdraw,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def aircraft(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        Aircraft,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def cubirds(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        CuBirds,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def dtd(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        Dtd,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def vggflower(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        VggFlower,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def trafficsigns(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        VggFlower,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def fungi(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        VggFlower,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def mscoco(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        VggFlower,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def ilsvrc2012(folder, shots=None, ways=None, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    defaults = {
        "transform": Compose([Resize((28, 28)), ToTensor()]),
        "class_augmentations": [Rotation([90, 180, 270])],
    }

    return helper_with_default(
        VggFlower,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def miniimagenet(
    folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs
):
    """Helper function to create a meta-dataset for the Mini-Imagenet dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `miniimagenet` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `MiniImagenet` class.

    See also
    --------
    `datasets.MiniImagenet` : Meta-dataset for the Mini-Imagenet dataset.
    """
    defaults = {"transform": Compose([Resize(84), ToTensor()])}

    return helper_with_default(
        MiniImagenet,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def tieredimagenet(
    folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs
):
    """Helper function to create a meta-dataset for the Tiered-Imagenet dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `tieredimagenet` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `TieredImagenet` class.

    See also
    --------
    `datasets.TieredImagenet` : Meta-dataset for the Tiered-Imagenet dataset.
    """
    defaults = {"transform": Compose([Resize(84), ToTensor()])}

    return helper_with_default(
        TieredImagenet,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def cifar_fs(folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the CIFAR-FS dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `cifar100` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `CIFARFS` class.

    See also
    --------
    `datasets.cifar100.CIFARFS` : Meta-dataset for the CIFAR-FS dataset.
    """
    return helper_with_default(
        CIFARFS,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults={},
        **kwargs
    )


def fc100(folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the FC100 dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `cifar100` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `FC100` class.

    See also
    --------
    `datasets.cifar100.FC100` : Meta-dataset for the FC100 dataset.
    """
    return helper_with_default(
        FC100,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults={},
        **kwargs
    )


def cub(folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Caltech-UCSD Birds dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `cub` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `CUB` class.

    See also
    --------
    `datasets.cub.CUB` : Meta-dataset for the Caltech-UCSD Birds dataset.
    """
    image_size = 84
    defaults = {
        "transform": Compose(
            [Resize(int(image_size * 1.5)), CenterCrop(image_size), ToTensor()]
        )
    }

    return helper_with_default(
        CUB,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )


def doublemnist(
    folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs
):
    """Helper function to create a meta-dataset for the Double MNIST dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `doublemnist` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `DoubleMNIST` class.

    See also
    --------
    `datasets.doublemnist.DoubleMNIST` : Meta-dataset for the Double MNIST dataset.
    """
    return helper_with_default(
        DoubleMNIST,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults={},
        **kwargs
    )


def triplemnist(
    folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs
):
    """Helper function to create a meta-dataset for the Triple MNIST dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `triplemnist` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `TripleMNIST` class.

    See also
    --------
    `datasets.triplemnist.TripleMNIST` : Meta-dataset for the Triple MNIST dataset.
    """
    return helper_with_default(
        TripleMNIST,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults={},
        **kwargs
    )


def pascal5i(folder, shots, ways=1, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the PASCAL-VOC dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification. Only supports 1-way currently

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    """
    defaults = {"transform": SegmentationPairTransform(500), "class_augmentations": []}
    return helper_with_default(
        Pascal5i,
        folder,
        shots,
        ways,
        shuffle=shuffle,
        test_shots=test_shots,
        seed=seed,
        defaults=defaults,
        **kwargs
    )
