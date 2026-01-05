import cv2
# import dlib
import torch, torchvision


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from align.alignment import alignment


def get_detectron_masks(image, predictor, classes=None, expansion=(1., 1.), debug=False):

    # run detectron
    H, W, C = image.shape

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    labels = instances.pred_classes.numpy()
    seg_masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    # get masks
    if classes is not None:
        indices = [i for i in range(len(instances)) if labels[i] in classes]
        seg_masks = seg_masks[indices]

    ew, eh = expansion
    boxes = np.round(boxes).astype(int)
    box_masks = np.zeros([len(instances), H, W], dtype=bool)

    for i in range(len(instances)):

        if classes is not None and labels[i] not in classes: continue

        x1, y1, x2, y2 = boxes[i]

        width = x2 - x1
        height = y2 - y1
        dw = int(round((ew - 1.) * width / 2.))
        dh = int(round((eh - 1.) * height / 2.))

        x1 = max(0, x1 - dw)
        x2 = min(W - 1, x2 + dw)
        y1 = max(0, y1 - dh)
        y2 = min(W - 1, y2 + dh)

        box_masks[i, y1:y2, x1:x2] = True

    return box_masks, seg_masks


def get_dlib_masks(image, detector, expansion=(2, 1.5)):

    H, W, C = image.shape
    eh, ew = expansion

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 0)
    mask = np.zeros([len(rects), H, W], dtype=bool)

    for i, rect in enumerate(rects):
        if hasattr(rect, "rect"):
            rect = rect.rect
        width = rect.right() - rect.left()
        height = rect.bottom() - rect.top()
        dw = int(round((ew - 1.) * width / 2.))
        dh = int(round((eh - 1.) * height / 2.))

        x1 = max(0, rect.left() - dw)
        x2 = min(W-1, rect.right() + dw)
        y1 = max(0, rect.top() - dh)
        y2 = min(W-1, rect.bottom() + dh)

        mask[i, y1:y2, x1:x2] = True

    return mask


def get_overlay_mask(image, mask, weight=0.3):

    mask = 255. * np.expand_dims(mask, axis=-1).astype(np.float32)
    mask = np.pad(mask, ((0, 0), (0, 0), (1, 1)), "constant")
    out = weight * image + (1 - weight) * mask
    out = np.round(out).astype(np.uint8)
    return out


def get_face_masks(image,
                   cfg_name="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                   dat_path="data/mmod_human_face_detector.dat",
                   predictor=None):
    if predictor is None:
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
        predictor = DefaultPredictor(cfg)

    # dlib_detector = dlib.get_frontal_face_detector()  # 效果最好且快
    # dlib_detector = dlib.cnn_face_detection_model_v1(dat_path)  # 效果好但是超级慢 （阻塞后在C++层面在跑）
    box_masks, seg_masks = get_detectron_masks(image, predictor, classes=[0])  # 效果一般会有裂口 但是快

    seg_mask = seg_masks.sum(axis=0) > 0
    # box_masks = get_dlib_masks(image, dlib_detector)

    return seg_mask, box_masks

def get_label_masks(image, left_label, right_label):

    aligned_left, LH, LA = alignment(image, left_label, image)
    aligned_right, RH, RA = alignment(image, right_label, image)
    combined_lr = cv2.addWeighted(aligned_left, 1, aligned_right, 1, 0)
    gray_lr = cv2.cvtColor(combined_lr, cv2.COLOR_RGB2GRAY)
    h, w = gray_lr.shape
    left_label = gray_lr[:, :w//2]
    right_label = gray_lr[:, w//2:]
    left_label = np.pad(left_label, ((0, 0), (0, w//2)), mode="constant", constant_values=0).astype(bool)
    right_label = np.pad(right_label, ((0, 0), (w//2, 0)), mode="constant", constant_values=0).astype(bool)
    box_masks = np.stack((left_label, right_label))
    seg_mask = gray_lr.astype(bool)

    seg_mask, _ = get_face_masks(image)

    return seg_mask, box_masks


def get_object_masks(image, classes,
                   cfg_name="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                   dat_path="data/mmod_human_face_detector.dat",
                   predictor=None):
    if predictor is None:
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
        predictor = DefaultPredictor(cfg)


    # box_masks, seg_masks = get_detectron_masks(image, predictor, classes)
    # seg_mask = seg_masks.sum(axis=0) > 0

    # seg_mask = ((seg_mask - 1) * -1).astype(bool)

    h, w = image.shape[:2]
    seg_mask = np.ones((h, w), dtype=bool)
    box_masks = np.expand_dims(seg_mask, axis=0)



    # sw = int(w / 3)
    # ew = int(w / 3 * 2)
    # seg_mask[:,sw:ew] = False


    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(seg_mask)
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(box_masks[0])
    # plt.colorbar()
    # plt.show()

    # plt.figure(figsize=(10, 8))
    # for i in range(5):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(box_masks[i])
    #     plt.colorbar()
    # plt.show()

    # box_masks = np.expand_dims(box_masks[0], axis=0)

    return seg_mask, box_masks



if __name__ == "__main__":

    wide = cv2.imread('./input/img.png')
    left = cv2.imread('./input/left1.png')
    right = cv2.imread('./input/right1.png')
    seg, box = get_label_masks(wide, left, right)

    plt.figure(figsize=(10, 10))
    plt.imshow(seg)
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.imshow(box[0])
    plt.imshow(box[1])
    plt.show()

