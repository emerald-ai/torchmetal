from torchmetal.datasets.cifar100 import CIFARFS, FC100
from torchmetal.datasets.cub import CUB
from torchmetal.datasets.doublemnist import DoubleMNIST
from torchmetal.datasets.miniimagenet import MiniImagenet
from torchmetal.datasets.omniglot import Omniglot
from torchmetal.datasets.pascal5i import Pascal5i
from torchmetal.datasets.tcga import TCGA
from torchmetal.datasets.tieredimagenet import TieredImagenet
from torchmetal.datasets.triplemnist import TripleMNIST

__all__ = [
    "TCGA",
    "Omniglot",
    "MiniImagenet",
    "TieredImagenet",
    "CIFARFS",
    "FC100",
    "CUB",
    "DoubleMNIST",
    "TripleMNIST",
    "Pascal5i",
]
