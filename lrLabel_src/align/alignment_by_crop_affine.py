import matplotlib.pyplot as plt
import numpy
import torch
from lightglue import viz2d
from lightglue.utils import load_image, rbd
from pathlib import Path
import numpy as np
import cv2

from alignment_utils import four_vertex_crop, get_human_mask, match_feature_point, feature_point_crop

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


image_left = '../images/left1.png'
image_right = '../images/right1.png'
image_wide = '../images/wide1.png'

left = cv2.imread(image_left)
right = cv2.imread(image_right)
wide = cv2.imread(image_wide)




src_pts, dst_pts = match_feature_point(load_image(Path(image_left)), load_image(Path(image_wide)))
axes = viz2d.plot_images([cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
viz2d.plot_matches(src_pts, dst_pts, color="lime", lw=0.2)

left_crop = feature_point_crop(left, src_pts)
plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB))
src_pts, dst_pts = match_feature_point(numpy_image_to_torch(left_crop), load_image(Path(image_wide)))

human_mask = get_human_mask(left_crop)
masked_src_pts = []
masked_dst_pts = []
for idx in range(src_pts.shape[0]):
    if not human_mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
        masked_src_pts.append(src_pts[idx])
        masked_dst_pts.append(dst_pts[idx])
masked_src_pts = np.array(masked_src_pts)
masked_dst_pts = np.array(masked_dst_pts)
H_l2w, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
H_w2l, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1.0)
A_l2w, mask1 = cv2.estimateAffinePartial2D(masked_src_pts, masked_dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)

lp = masked_src_pts[np.array(mask1).squeeze() == 1]
wp = masked_dst_pts[np.array(mask1).squeeze() == 1]
axes = viz2d.plot_images([cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
viz2d.plot_matches(lp, wp, color="lime", lw=0.2)

src_pts, dst_pts = match_feature_point(load_image(Path(image_right)), load_image(Path(image_wide)))

right_crop = feature_point_crop(right, src_pts)
src_pts, dst_pts = match_feature_point(numpy_image_to_torch(right_crop), load_image(Path(image_wide)))

human_mask = get_human_mask(right_crop)
masked_src_pts = []
masked_dst_pts = []
for idx in range(src_pts.shape[0]):
    if not human_mask[int(src_pts[idx][1]), int(src_pts[idx][0])]:
        masked_src_pts.append(src_pts[idx])
        masked_dst_pts.append(dst_pts[idx])
masked_src_pts = np.array(masked_src_pts)
masked_dst_pts = np.array(masked_dst_pts)
# H_r2w, _ = cv2.findHomography(masked_src_pts, masked_dst_pts, cv2.RANSAC, 1.0)
# H_w2r, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1.0)
A_r2w, mask2 = cv2.estimateAffinePartial2D(masked_src_pts, masked_dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
lp = masked_src_pts[np.array(mask2).squeeze() == 1]
wp = masked_dst_pts[np.array(mask2).squeeze() == 1]
axes = viz2d.plot_images([cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB), cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)])
viz2d.plot_matches(lp, wp, color="lime", lw=0.2)

# w2l and w2r
# h, w = left.shape[:2]
# temp_w2l = cv2.warpPerspective(wide, H_w2l, (w, h))
# h, w = right.shape[:2]
# temp_w2r = cv2.warpPerspective(wide, H_w2r, (w, h))
# plt.figure(figsize=(30, 20))
# plt.subplot(1,3,1)
# plt.imshow(cv2.cvtColor(temp_w2l, cv2.COLOR_BGR2RGB))
# plt.title('temp_w2l')
# plt.subplot(1,3,2)
# plt.imshow(cv2.cvtColor(wide, cv2.COLOR_BGR2RGB))
# plt.title('temp_w')
# plt.subplot(1,3,3)
# plt.imshow(cv2.cvtColor(temp_w2r, cv2.COLOR_BGR2RGB))
# plt.title('temp_w2')


h, w = wide.shape[:2]
left_warp = cv2.warpAffine(left_crop, A_l2w, (w, h))
# left_warp = cv2.warpPerspective(left, H_l2w, (w, h))

# 裁剪出图像
gray = cv2.cvtColor(left_warp, cv2.COLOR_BGR2GRAY)
mask = gray > 0  # 非黑色区域为 True
coords = np.column_stack(np.where(mask))
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
cutting_left_warp = left_warp[y_min:y_max, x_min:x_max]

h, w = wide.shape[:2]
right_warp = cv2.warpAffine(right_crop, A_r2w, (w, h))
# right_warp = cv2.warpPerspective(right, H_r2w, (w, h))

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

plt.tight_layout()
plt.show()


combined_wide = cv2.addWeighted(left_warp, 1, right_warp, 1, 0)
combined_wide = cv2.addWeighted(combined_wide, 0.6, wide, 0.4, 0)
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(combined_wide, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Affine Transform', size=30)
plt.tight_layout()
plt.show()
# cv2.namedWindow("combined_wide", cv2.WINDOW_NORMAL)
# cv2.imshow('combined_wide', combined_wide)
# cv2.waitKey(0)
