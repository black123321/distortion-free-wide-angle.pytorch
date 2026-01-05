from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
import torch
import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def four_vertex_crop(img):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取所有非黑像素（四边形区域）
    ys, xs = np.where(mask > 0)
    points = np.stack((xs, ys), axis=1)  # N×2 的点集，每行是 (x, y)

    # 计算四个角点
    sums = points[:, 0] + points[:, 1]       # x + y
    diffs = points[:, 0] - points[:, 1]      # x - y

    left_top = points[np.argmin(sums)]
    right_bottom = points[np.argmax(sums)]
    right_top = points[np.argmax(diffs)]
    left_bottom = points[np.argmin(diffs)]

    pts = np.array([left_top, right_top, right_bottom, left_bottom])

    xs = pts[:, 0]
    ys = pts[:, 1]

    sorted_xs = np.sort(xs)
    sorted_ys = np.sort(ys)

    crop_img = img[sorted_ys[1]:sorted_ys[2], sorted_xs[1]:sorted_xs[2]]

    return crop_img

def feature_point_crop(img, src_pts):
    xmin = int(np.min(src_pts[:, 0]))
    xmax = int(np.max(src_pts[:, 0]))
    ymin = int(np.min(src_pts[:, 1]))
    ymax = int(np.max(src_pts[:, 1]))

    crop_img = img[ymin:ymax, xmin:xmax]

    return crop_img


def get_human_mask(image):
    cfg = get_cfg()
    cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    predictor = DefaultPredictor(cfg)


    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    seg_masks = instances.pred_masks.numpy()
    labels = instances.pred_classes.numpy()
    classes = [0] # mask label of human
    indices = [i for i in range(len(instances)) if labels[i] in classes]
    seg_masks = seg_masks[indices]
    seg_mask = seg_masks.sum(axis=0) > 0
    return seg_mask

def get_src_point_inout_mask(src_pts, dst_pts, mask, in_mask):
    masked_src_pts = []
    masked_dst_pts = []
    if in_mask:
        for idx in range(src_pts.shape[0]):
            if mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
                masked_src_pts.append(src_pts[idx])
                masked_dst_pts.append(dst_pts[idx])
    elif not in_mask:
        for idx in range(src_pts.shape[0]):
            if not mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
                masked_src_pts.append(src_pts[idx])
                masked_dst_pts.append(dst_pts[idx])
    masked_src_pts = np.array(masked_src_pts)
    masked_dst_pts = np.array(masked_dst_pts)
    return masked_src_pts, masked_dst_pts


def match_feature_point(img0, img1):
    if isinstance(img0, np.ndarray):
        img0 = numpy_image_to_torch(img0)
    if isinstance(img1, np.ndarray):
        img1 = numpy_image_to_torch(img1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)
    feats0 = extractor.extract(img0.to(device))
    feats1 = extractor.extract(img1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    m_kpts0 = m_kpts0.cpu().numpy()
    m_kpts1 = m_kpts1.cpu().numpy()

    return m_kpts0, m_kpts1
