import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from pathlib import Path
import torch
import numpy as np
import cv2

from detectron2.config import get_cfg

from my_src.align.alignment_utils import four_vertex_crop, get_human_mask, match_feature_point, feature_point_crop, get_src_point_inout_mask


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def alignment(src_img, dst_img, input_img, affine=True):
    #resize label
    h, w = src_img.shape[:2]
    dh, dw = dst_img.shape[:2]
    resized_dst_img = cv2.resize(dst_img, (int(dw * h / dh), h))
    # resized_dst_img = dst_img

    src_pts, dst_pts = match_feature_point(numpy_image_to_torch(src_img), numpy_image_to_torch(resized_dst_img))
    H_s2d, m1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)
    # visualization of matched feature points
    # sp = src_pts[np.array(m1).squeeze() == 1]
    # dp = dst_pts[np.array(m1).squeeze() == 1]
    # axes = viz2d.plot_images([cv2.cvtColor(resized_dst_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)])
    # viz2d.plot_matches(dp, sp, color="lime", lw=0.2)

    # source image to destination image
    h, w = resized_dst_img.shape[:2]
    aligned_img = cv2.warpPerspective(input_img, H_s2d, (w, h))

    A_a2s = None
    if affine:
        h, w = src_img.shape[:2]

        src_pts, dst_pts = match_feature_point(numpy_image_to_torch(aligned_img), numpy_image_to_torch(src_img))
        # axes = viz2d.plot_images([cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)])
        # viz2d.plot_matches(src_pts, dst_pts, color="lime", lw=0.2)

        # human_mask = get_human_mask(aligned_img)
        # masked_src_pts, masked_dst_pts = get_src_point_inout_mask(src_pts, dst_pts, human_mask, in_mask=True)
        # A_a2s, m1 = cv2.estimateAffinePartial2D(masked_dst_pts, masked_src_pts, method=cv2.RANSAC,
        #                                          ransacReprojThreshold=20.0)
        A_a2s, m1 = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                 ransacReprojThreshold=30.0)
        # visualization of matched feature points in human body
        # axes = viz2d.plot_images([cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)])
        # lp = src_pts[np.array(m1).squeeze() == 1]
        # wp = dst_pts[np.array(m1).squeeze() == 1]
        # viz2d.plot_matches(lp, wp, color="lime", lw=0.2)

        aligned_img = cv2.warpAffine(aligned_img, A_a2s, (w, h))

    return aligned_img, H_s2d, A_a2s