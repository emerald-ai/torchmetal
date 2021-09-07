# coding=utf-8
# Copyright 2021 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
# pyformat: disable
r"""Main file for converting the datasets used in the benchmark into records.
Example command to convert dataset omniglot:
    # pylint: disable=line-too-long
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
        --dataset=omniglot \
        --omniglot_data_root=<path/to/omniglot> \
        --records_root=<path/to/records> \
        --splits_root=<path/to/splits>
    # pylint: enable=line-too-long
    """
# pyformat: enable

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import json

from torchmetal.datasets.metadataset import dataset_to_hdf5
from omegaconf import OmegaConf
import h5py as h5

# TODO: combine all h5 records into a train, test, val format based on the
# splits file in dataset folder, if none random based on number defined at top
# of records files
#
#def with_splits(splits, records):
#    hftrain = h5.file('train.h5')
#    hftest = h5.file('test.h5')
#    hfval = h5.file('val.h5')
#
#def without_splits(records):
#
#def combine_h5(dataset_name):
#    with open(f'./{dataset_name}/splits/{dataset_name}_splits.json') as f:
#        splits = json.loads(f)
#
#    if 'train' in splits and 'test' in splits and 'val' in splits:



def main():


    cfg = OmegaConf.load('./config/config.yaml')
    for item, value in cfg.data_set.items():
        if value["name"] == cfg.dataset:
            converter_args = value



    if converter_args is None:
        raise ValueError(f"Dataset {cfg.dataset}'s config not found")
    else:
        if cfg.dataset == 'omniglot':
            converter =  dataset_to_hdf5.OmniglotConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root')
                                                          )
        elif cfg.dataset == 'aircraft':
            converter = dataset_to_hdf5.AircraftConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                         )
        elif cfg.dataset == 'cu_birds':
            converter = dataset_to_hdf5.CUBirdsConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                        )
        elif cfg.dataset == 'dtd':
            converter = dataset_to_hdf5.DTDconverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                    )
        elif cfg.dataset == 'quickdraw':
            converter = dataset_to_hdf5.QuickdrawConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                          )
        elif cfg.dataset == 'fungi':
            converter = dataset_to_hdf5.FungiConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                      )
        elif cfg.dataset == 'vgg_flower':
            converter = dataset_to_hdf5.VGGFlowerConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                          )
        elif cfg.dataset == 'traffic_sign':
            converter = dataset_to_hdf5.TrafficSignConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                            )
        elif cfg.dataset == 'mscoco':
            converter = dataset_to_hdf5.MSCOCOConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                       )
        elif cfg.dataset == 'mini_imagenet':
            converter =  dataset_to_hdf5.MiniImageNetConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                              )
        elif cfg.dataset == 'ilsvrc_2012':
            converter = dataset_to_hdf5.ImageNetConverter(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                         )
        elif cfg.dataset == 'ilsvrc_2012_v2':
            converter = dataset_to_hdf5.ImageNetConverterV2(
                                        cfg.dataset,
                                        converter_args.get('data_root'),
                                        converter_args.get('splits_root'),
                                        converter_args.get('records_root'),
                                                             )
        if cfg.dataset == 'mini_imagenet':
            # MiniImagenet is for diagnostics purposes only,
            # do not use the default records_path to avoid confusion.
            records_path = cfg.mini_imagenet_records_dir

            logging.info(
                'Creating %s specification and records in directory %s...',
                cfg.dataset, records_path)

        else:
            records_path = None
            logging.info(
                'Creating %s specification and records',
                cfg.dataset)
            converter.convert_dataset()


if __name__ == '__main__':
    main()
