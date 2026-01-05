import matplotlib.pyplot as plt
import numpy
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from pathlib import Path
import torch
import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from alignment_utils import four_vertex_crop, get_human_mask, match_feature_point, feature_point_crop, get_src_point_inout_mask

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


if __name__ == "__main__":
    cfg = get_cfg()
    image_left = '../images/left1.png'
    image_right = '../images/right1.png'
    image_wide = '../images/wide1.png'

    left = cv2.imread(image_left)
    right = cv2.imread(image_right)
    wide = cv2.imread(image_wide)

    h, w = wide.shape[:2]

    lh, lw = left.shape[:2]

    # label resize后背景更贴合原图，人脸校正变差
    resized_left = cv2.resize(left, (int(lw*h/lh), h))
    resized_right = cv2.resize(right, (int(lw*h/lh), h))

    # 原label 人脸校正效果好，背景不贴合
    # resized_left = left
    # resized_right = right


    src_pts, dst_pts = match_feature_point(load_image(Path(image_wide)), numpy_image_to_torch(resized_left))
    H_w2l, m1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)
    sp = src_pts[np.array(m1).squeeze() == 1]
    dp = dst_pts[np.array(m1).squeeze() == 1]
    # axes = viz2d.plot_images([cv2.cvtColor(wide, cv2.COLOR_BGR2RGB), cv2.cvtColor(resized_left, cv2.COLOR_BGR2RGB)])
    # viz2d.plot_matches(sp, dp, color="lime", lw=0.2)

    src_pts, dst_pts = match_feature_point(load_image(Path(image_wide)), numpy_image_to_torch(resized_right))
    H_w2r, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)
    # axes = viz2d.plot_images([cv2.cvtColor(wide, cv2.COLOR_BGR2RGB), cv2.cvtColor(resized_right, cv2.COLOR_BGR2RGB)])
    # viz2d.plot_matches(src_pts, dst_pts, color="lime", lw=0.2)

    # w2l and w2r
    h, w = resized_left.shape[:2]
    temp_w2l = cv2.warpPerspective(wide, H_w2l, (w, h))

    h, w = resized_right.shape[:2]
    temp_w2r = cv2.warpPerspective(wide, H_w2r, (w, h))



    plt.figure(figsize=(30, 20))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(temp_w2l, cv2.COLOR_BGR2RGB))
    plt.title('temp_w2l')
    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(wide, cv2.COLOR_BGR2RGB))
    plt.title('temp_w')
    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(temp_w2r, cv2.COLOR_BGR2RGB))
    plt.title('temp_w2r')
    plt.tight_layout()

    h, w = wide.shape[:2]


    # src_pts, dst_pts = match_feature_point(numpy_image_to_torch(temp_w2l), numpy_image_to_torch(resized_left))
    # crop_temp_w2l = feature_point_crop(temp_w2l, src_pts)
    # cv2.imwrite('./result/feature_combined_left.png', combined)
    src_pts, dst_pts = match_feature_point(numpy_image_to_torch(temp_w2l), load_image(Path(image_wide)))
    human_mask = get_human_mask(temp_w2l)

    masked_src_pts, masked_dst_pts = get_src_point_inout_mask(src_pts, dst_pts, human_mask, in_mask=True)
    A_wl2l, m1 = cv2.estimateAffinePartial2D(masked_src_pts, masked_dst_pts, method=cv2.RANSAC, ransacReprojThreshold=20.0)
    axes = viz2d.plot_images([cv2.cvtColor(temp_w2l, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
    lp = masked_src_pts[np.array(m1).squeeze() == 1]
    wp = masked_dst_pts[np.array(m1).squeeze() == 1]
    viz2d.plot_matches(lp, wp, color="lime", lw=0.2)
    crop_temp_w2l = cv2.warpAffine(temp_w2l, A_wl2l, (w, h))


    # src_pts, dst_pts = match_feature_point(numpy_image_to_torch(temp_w2r), numpy_image_to_torch(resized_right))
    # crop_temp_w2r = feature_point_crop(temp_w2r, src_pts)
    # cv2.imwrite('./result/feature_combined_right.png', combined)


    # combined_right = cv2.addWeighted(temp_w2r, 0.6, resized_right, 0.6, 0)
    # combined_left = cv2.addWeighted(temp_w2l, 0.6, resized_left, 0.6, 0)
    # plt.figure(figsize=(20, 10))
    # plt.subplot(1,2,1)
    # plt.imshow(cv2.cvtColor(combined_left, cv2.COLOR_BGR2RGB))
    # plt.subplot(1,2,2)
    # plt.imshow(cv2.cvtColor(combined_right, cv2.COLOR_BGR2RGB))
    # plt.tight_layout()


    src_pts, dst_pts = match_feature_point(numpy_image_to_torch(temp_w2r), load_image(Path(image_wide)))
    human_mask = get_human_mask(temp_w2r)
    masked_src_pts = []
    masked_dst_pts = []
    for idx in range(src_pts.shape[0]):
        if not human_mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
            masked_src_pts.append(src_pts[idx])
            masked_dst_pts.append(dst_pts[idx])
    masked_src_pts = np.array(masked_src_pts)
    masked_dst_pts = np.array(masked_dst_pts)
    A_wr2r, m2 = cv2.estimateAffinePartial2D(masked_src_pts, masked_dst_pts, method=cv2.RANSAC, ransacReprojThreshold=20.0)
    # axes = viz2d.plot_images([cv2.cvtColor(temp_w2r, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
    # lp = masked_src_pts[np.array(m2).squeeze() == 1]
    # wp = masked_dst_pts[np.array(m2).squeeze() == 1]
    # viz2d.plot_matches(lp, wp, color="lime", lw=0.2)
    crop_temp_w2r = cv2.warpAffine(temp_w2r, A_wr2r, (w, h))


    # plt.figure(figsize=(30, 20))
    # plt.subplot(1,3,1)
    # plt.imshow(cv2.cvtColor(crop_temp_w2l, cv2.COLOR_BGR2RGB))
    # plt.title('temp_w2l')
    # plt.subplot(1,3,2)
    # plt.imshow(cv2.cvtColor(wide, cv2.COLOR_BGR2RGB))
    # plt.title('temp_w')
    # plt.subplot(1,3,3)
    # plt.imshow(cv2.cvtColor(crop_temp_w2r, cv2.COLOR_BGR2RGB))
    # plt.title('temp_w2r')
    # plt.show()


    plt.figure(figsize=(20, 10))
    combined_wide = cv2.addWeighted(crop_temp_w2l, 1, crop_temp_w2r, 1, 0)
    cv2.imwrite('../result/combined_wide.png', combined_wide)
    combined_wide = cv2.addWeighted(combined_wide, 0.6, wide, 0.6, 0)
    plt.imshow(cv2.cvtColor(combined_wide, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # cv2.namedWindow('combined_wide', cv2.WINDOW_NORMAL)
    # cv2.imshow('combined_wide', combined_wide)
    # cv2.waitKey(0)