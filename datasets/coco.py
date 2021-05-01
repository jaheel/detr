# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        """construct dataset

        Parameters
        ----------
        img_folder : {str-like, scalar}, the img folder path

        ann_file : {str-like, scalar}, the annotation file path

        transforms : {function_name_list, vector}, the image transform operation list

        return_masks : {boolean, scalar}, whether segmentation
        """
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        """

        Parameters
        ----------
        idx : {int, scalar} the image index
        """

        # img : {PIL.Image.Image, model=RGB, size=(W, H)}
        # target : {dict_list}
        #          one of them : {"bbox" : [x, y, w, h],
        #                           "area" : box_area,
        #                           ...}
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx] # get the image id
        target = {'image_id': image_id, 'annotations': target}

        # -----------------
        # image prepare
        #
        # img : {PIL.Image.Image, model=RGB, size=(W, H)}
        
        #target : {dict_list}
        #           {
        #               "boxes" : {float, matrix} shape of (target_bbox_number, [x_1, y_1, x_2, y_2]), target bbox coordinate set
        #               "labels" : {int, vector} shape of (target_bbox_number), target bbox label class
        #               "image_id" : {int, scalar}, image id
        #               "area" : {float, vector} shape of (target_bbox_number), every bbox area
        #               "iscrowd" : {int, vector}, value is (0 or 1), shape of (target_bbox_number), 0: segmentation is polygon format, 1 : segmentation is RLE format
        #               "orig_size" : {int} of [H, W]
        #               "size" : {int} of [H, W]
        #           }
        img, target = self.prepare(img, target)
        # -----------------

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        """

        Parameters
        ----------
        image : {PIL.Image.Image, model=RGB, size=(W, H)}

        target : {dict_list}
                    {
                        "image_id" : image id,
                        "annotations" : bbox_annotation list
                    }
                    one of bbox_annotation : {
                                    "segmentation": [...],
                                    "bbox" : [x, y, w, h],
                                    "area" : box_area,
                                    "iscrowd" : int_scalar,
                                    "image_id" : image id,
                                    "category_id" : int_scalar,
                                    "id" : int_scalar,
                                    }
        
        Returns
        -------
        image : {PIL.Image.Image, model=RGB, size=(W, H)}
        
        target : {dict_list}
                    {
                        "boxes" : {float, matrix} shape of (target_bbox_number, [x_1, y_1, x_2, y_2]), target bbox coordinate set
                        "labels" : {int, vector} shape of (target_bbox_number), target bbox label class
                        "image_id" : {int, scalar}, image id
                        "area" : {float, vector} shape of (target_bbox_number), every bbox area
                        "iscrowd" : {int, vector}, value is (0 or 1), shape of (target_bbox_number), 0: segmentation is polygon format, 1 : segmentation is RLE format
                        "orig_size" : {int} of [H, W]
                        "size" : {int} of [H, W]
                    }

        """
        w, h = image.size

        # --------------------
        # change image_id to tensor format
        #
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        # --------------------

        anno = target["annotations"]

        # --------------------
        # filter crowd
        #
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # --------------------

        # --------------------
        # set the boxes
        # 
        # boxes : (target_bbox_number, 4)
        #
        boxes = [obj["bbox"] for obj in anno]
        # --------------------

        # --------------------
        # guard against no boxes via resizing
        #
        # boxes change to tensor format
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # (x, y, w, h) -> (x_lr, y_lr, x_rd, y_rd)
        boxes[:, 2:] += boxes[:, :2]
        
        # set the x, y upper limit and lower limit
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # --------------------

        # --------------------
        # get the obj class in bbox
        # 
        # classes : shape of (target_bbox_number)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        # --------------------

        # --------------------
        # whether add the segmentation data
        #
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        # --------------------

        # --------------------
        # keypoints
        #
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        # --------------------

        # --------------------
        # find x_2 > x_1, y_2 > y_1
        #
        # keep : {boolean, vector} of shape (target_bbox_number)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        

        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        # --------------------

        # --------------------
        # per image's target setting
        #
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        
        # optial (mask, keypoints)
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # --------------------

        return image, target


def make_coco_transforms(image_set):
    """coco dataset preprocessing

    Parameters
    ----------
    image_set : {str-like, scalar} "train" or "val"

    Returns
    -------
    result : {T.Compose list} the image transform operation list
    """
    # -------------------
    # normalize the image
    #
    normalize = T.Compose([
        # processing method
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # R,G,B每层的归一化用到的均值和方差
    ])
    # -------------------

    # -------------------
    #
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # -------------------


    # -------------------
    # according to the image set types,
    #   determine the preprocessing steps of the image
    #
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    # -------------------

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    """build dataset
    
    Parameters
    ----------
    image_set : {str-like, scalar} "train" or "val"

    args : command args parameter
    """
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
