"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.video_text_pair_datasets import VideoTextPairDataset
from lavis.datasets.datasets.video_text_pair_datasets import ImageBindVideoTextPairDataset


@registry.register_builder("webvid_2m")
class WebVid2MBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/defaults_2m.yaml"
    }


@registry.register_builder("webvid_10m")
class WebVid10MBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/defaults_10m.yaml"
    }


@registry.register_builder("webvid_2m_imagebind")
class WebVid2MImageBindBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageBindVideoTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/imagebind_2m.yaml"
    }

@registry.register_builder("webvid_2m_msrvtt")
class WebVid2MBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/synthetic_msrvtt_2m.yaml"
    }