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
    seg_mask = seg_masks.sum(axis=0) > 0
    return seg_mask


def match_feature_point(img0, img1):
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


image_left = './images/left2.png'
image_right = './images/right2.png'
image_wide = './images/wide2.png'

left = cv2.imread(image_left)
right = cv2.imread(image_right)
wide = cv2.imread(image_wide)

src_pts, dst_pts = match_feature_point(load_image(Path(image_left)), load_image(Path(image_wide)))
axes = viz2d.plot_images([cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
viz2d.plot_matches(src_pts, dst_pts, color="lime", lw=0.2)

human_mask = get_human_mask(left)
masked_src_pts = []
masked_dst_pts = []
for idx in range(src_pts.shape[0]):
    if not human_mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
        masked_src_pts.append(src_pts[idx])
        masked_dst_pts.append(dst_pts[idx])
masked_src_pts = np.array(masked_src_pts)
masked_dst_pts = np.array(masked_dst_pts)
H_l2w, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
lp = src_pts[np.array(mask1).squeeze() == 1]
wp = dst_pts[np.array(mask1).squeeze() == 1]
axes = viz2d.plot_images([cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
viz2d.plot_matches(lp, wp, color="lime", lw=0.2)

src_pts, dst_pts = match_feature_point(load_image(Path(image_right)), load_image(Path(image_wide)))
human_mask = get_human_mask(right)
masked_src_pts = []
masked_dst_pts = []
for idx in range(src_pts.shape[0]):
    if not human_mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
        masked_src_pts.append(src_pts[idx])
        masked_dst_pts.append(dst_pts[idx])
masked_src_pts = np.array(masked_src_pts)
masked_dst_pts = np.array(masked_dst_pts)
H_r2w, _ = cv2.findHomography(masked_src_pts, masked_dst_pts, cv2.RANSAC, 1.0)



h, w = wide.shape[:2]
left_warp = cv2.warpPerspective(left, H_l2w, (w, h))
# 裁剪出图像
gray = cv2.cvtColor(left_warp, cv2.COLOR_BGR2GRAY)
mask = gray > 0  # 非黑色区域为 True
coords = np.column_stack(np.where(mask))
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
cutting_left_warp = left_warp[y_min:y_max, x_min:x_max]

h, w = wide.shape[:2]
right_warp = cv2.warpPerspective(right, H_r2w, (w, h))

gray = cv2.cvtColor(right_warp, cv2.COLOR_BGR2GRAY)
mask = gray > 0  # 非黑色区域为 True
coords = np.column_stack(np.where(mask))
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
cutting_right_warp = right_warp[y_min:y_max, x_min:x_max]


plt.figure(figsize=(30, 20))
plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
plt.title('left')

plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(wide, cv2.COLOR_BGR2RGB))
plt.title('wide')

plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
plt.title('right')

plt.subplot(2,3,4)
plt.imshow(cv2.cvtColor(cutting_left_warp, cv2.COLOR_BGR2RGB))
plt.title('left_warp')


plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(wide, cv2.COLOR_BGR2RGB))
plt.title('wide')

plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(cutting_right_warp, cv2.COLOR_BGR2RGB))
plt.title('right_warp')


combined_wide = cv2.addWeighted(left_warp, 1, right_warp, 1, 0)
combined_wide = cv2.addWeighted(cv2.cvtColor(combined_wide, cv2.COLOR_BGR2RGB), 0.6, cv2.cvtColor(wide, cv2.COLOR_BGR2RGB), 0.6, 0)
plt.figure(figsize=(30, 20))
plt.title('Homography Transform', size=60)
plt.axis('off')
plt.imshow(combined_wide)
 
plt.tight_layout()
plt.show()