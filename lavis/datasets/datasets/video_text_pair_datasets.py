"""
 Copyright (c) 2022, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
import warnings

from lavis.datasets.datasets.base_dataset import BaseDataset


class VideoTextPairDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # fetch video
        num_retries = 10  # skip error videos

        for _ in range(num_retries):

            ann = self.annotation[index]

            # vname = ann["image"].replace("dataset", "dataset_downsampled")
            vname = ann["image"]
            video_path = os.path.join(self.vis_root, vname)

            # read with retries
            for _ in range(3):
                try:
                    video = self.vis_processor(video_path)
                    caption = self.text_processor(ann["caption"])
                    break
                except:
                    video = None

            if video is None:
                warnings.warn(f"Failed to load examples with video: {video_path}. Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break

        if video is None:
            raise RuntimeError(f"Failed to fetch image/video after {num_retries} retries.")

        # return {"image": video, "image_id": vname, "text_input": caption}
        return {"image": video, "text_input": caption}


class ImageBindVideoTextPairDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # vname = ann["image"].replace("dataset", "dataset_downsampled")
        vname = ann["image"]
        video_path = os.path.join(self.vis_root, vname)

        video = video_path
        caption = self.text_processor(ann["caption"])

        return {"image": video, "text_input": caption}
