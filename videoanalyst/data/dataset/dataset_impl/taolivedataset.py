# -*- coding: utf-8 -*-
import contextlib
import io
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from loguru import logger
from pycocotools import mask as MaskApi
from pycocotools.coco import COCO

from videoanalyst.data.dataset.dataset_base import (TRACK_DATASETS,
                                                    VOS_DATASETS, DatasetBase)
from videoanalyst.pipeline.utils.bbox import xywh2xyxy


@TRACK_DATASETS.register
@VOS_DATASETS.register
class TaoliveDataset(DatasetBase):
    r"""
    COCO dataset helper
    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subsets: list
        dataset split name [train2017,val2017]
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    """
    data_items = []
    _DUMMY_ANNO = [[-1, -1, 0, 0]]

    default_hyper_params = dict(
        dataset_root="datasets/coco2017",
        subsets=[
            "val2017",
        ],
        ratio=1.0,
        with_mask=False,
    )

    def __init__(self) -> None:
        r"""
        Create dataset with config
        """
        super(TaoliveDataset, self).__init__()
        self._state["dataset"] = None
        self.data_items = []

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = self._hyper_params["dataset_root"]
        self._hyper_params["dataset_root"] = osp.realpath(dataset_root)
        if len(TaoliveDataset.data_items) == 0:
            self._ensure_cache()

    def __getitem__(self, item):
        """
        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        record = self.data_items[item]
        box_anno = record['annotations']
        if len(box_anno) <= 0:
            box_anno = self._DUMMY_ANNO
        # box_anno = xywh2xyxy(box_anno)
        box_anno = np.array(box_anno)
        sequence_data = dict(image=record["file_name"], anno=box_anno)
        return sequence_data

    def __len__(self):
        return len(self.data_items)

    def _ensure_cache(self):
        img_insid_to_img = {}
        dataset_root = self._hyper_params["dataset_root"]
        subsets = self._hyper_params["subsets"]
        with open(os.path.join(dataset_root, 'images.txt'), 'r') as f_img, \
                open(os.path.join(dataset_root, 'image_class_label.txt'), 'r') as f_label, \
                open(os.path.join(dataset_root, 'bounding_boxes.txt'), 'r') as f_box:
            # class_id = {}
            # img_path = []
            # boxs = []
            # labels = []
            # categories = []
            for line_img, line_label, line_box in zip(f_img, f_label, f_box):
                fname = os.path.join(dataset_root, line_img.strip().split()[-1])
                label = line_label.strip().split()[-1]
                box = [int(float(v)) for v in line_box.split()[1:5]]
                category = line_box.split()[5]

                # img_path.append(fname)
                # labels.append(label)
                # boxs.append(box)
                # categories.append(int(category))

                if label != '0':
                    if not img_insid_to_img.__contains__(label):
                        img_insid_to_img[label] = [dict(path=fname, box=box)]
                    else:
                        img_insid_to_img[label].append(dict(path=fname, box=box))

        v_insid_to_img = {}
        with open(os.path.join(dataset_root, 'images.txt'), 'r') as f_img, \
                open(os.path.join(dataset_root, 'image_class_label.txt'), 'r') as f_label, \
                open(os.path.join(dataset_root, 'bounding_boxes.txt'), 'r') as f_box:
            # class_id = {}
            # img_path = []
            # boxs = []
            # labels = []
            # categories = []
            for line_img, line_label, line_box in zip(f_img, f_label, f_box):
                fname = os.path.join(dataset_root, line_img.strip().split()[-1])
                label = line_label.strip().split()[-1]
                box = [int(float(v)) for v in line_box.split()[1:5]]
                category = line_box.split()[5]

                # img_path.append(fname)
                # labels.append(label)
                # boxs.append(box)
                # categories.append(int(category))

                if label != '0':
                    if not v_insid_to_img.__contains__(label):
                        v_insid_to_img[label] = [dict(path=fname, box=box)]
                    else:
                        v_insid_to_img[label].append(dict(path=fname, box=box))

        img_id = 0
        for ins_id in list(img_insid_to_img.keys()):
            if not v_insid_to_img.__contains__(ins_id):
                continue
            record = {}
            record["file_name"] = [
                os.path.join(dataset_root, img_insid_to_img[ins_id][0]['path']),
                os.path.join(dataset_root, v_insid_to_img[ins_id][0]['path'])
            ]

            record["annotations"] = [
                img_insid_to_img[ins_id][0]['box'],
                v_insid_to_img[ins_id][0]['box']
            ]
            self.data_items.append(record)
