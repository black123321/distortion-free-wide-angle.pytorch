import cv2
import numpy as np
import matplotlib.pyplot as plt
from align.alignment_utils import four_vertex_crop, get_human_mask, match_feature_point, feature_point_crop, get_src_point_inout_mask
import torch

np.set_printoptions(threshold=np.inf)
from my_src.align.alignment import alignment
from my_src.perception import get_label_masks
from my_src.visualization import get_overlay_flow


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

eps = 1e-6

def FOV2f(fov, diagnoal):

    f = diagnoal / (2 * np.tan(fov / 2))
    return f


def correct(image, fov):

    h, w, _ = image.shape
    d = min(h, w)
    f = FOV2f(fov, d)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))

    x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
    y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
    x, y = np.meshgrid(x, y, indexing="xy")

    coords = np.stack([x, y], axis=-1)
    rp = np.linalg.norm(coords, axis=-1)
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    x = x / ru * rp + w / 2
    y = y / ru * rp + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out


def get_uniform_stereo_mesh(image, left_label, right_label, Q, mesh_ds_ratio):
    H, W, _ = image.shape
    label_H, label_W, _ = left_label.shape

    image_ds = cv2.resize(image, (W // mesh_ds_ratio, H // mesh_ds_ratio))
    left_label_ds = cv2.resize(left_label, (label_W // mesh_ds_ratio, label_H // mesh_ds_ratio))
    right_label_ds = cv2.resize(right_label, (label_W // mesh_ds_ratio, label_H // mesh_ds_ratio))

    _, LH, LA = alignment(image_ds, left_label_ds, image_ds)
    _, RH, RA = alignment(image_ds, right_label_ds, image_ds)


    Hm = H // mesh_ds_ratio + 2 * Q
    Wm = W // mesh_ds_ratio + 2 * Q


    label_Hm = label_H // mesh_ds_ratio + 2 * Q
    label_Wm = label_W // mesh_ds_ratio + 2 * Q

    x = (np.arange(0, Wm, 1)).astype(np.float32) - (Wm // 2) + 0.5
    y = (np.arange(0, Hm, 1)).astype(np.float32) - (Hm // 2) + 0.5
    x = x * mesh_ds_ratio
    y = y * mesh_ds_ratio
    x, y = np.meshgrid(x, y, indexing="xy")

    mesh_uniform = np.stack([x, y], axis=0)



    x = (np.arange(0, Wm, 1)).astype(np.float32)
    y = (np.arange(0, Hm, 1)).astype(np.float32)
    x = x * mesh_ds_ratio
    y = y * mesh_ds_ratio
    x, y = np.meshgrid(x, y, indexing="xy")

    mesh_label = np.stack([x, y], axis=0)

    # S = [[1 / mesh_ds_ratio, 0, 0],
    #      [0, 1 / mesh_ds_ratio, 0],
    #      [0, 0, 1]]

    # LH = S * LH * np.linalg.inv(S)
    # RH = S * RH * np.linalg.inv(S)
    # LH = np.linalg.inv(LH)
    # RH = np.linalg.inv(RH)
    coords = np.moveaxis(mesh_label, 0, -1)
    coords_left = cv2.warpPerspective(coords, LH, (label_Wm, label_Hm))
    coords_right = cv2.warpPerspective(coords, RH, (label_Wm, label_Hm))

    # print(coords_left)
    # print(coords_right)

    # LA = np.vstack([LA, [0, 0, 1]])  # 扩展为 3×3
    # RA = np.vstack([RA, [0, 0, 1]])  # 扩展为 3×3
    # LA = S * LA * np.linalg.inv(S)
    # RA = S * RA * np.linalg.inv(S)
    # LA = LA[:2, :]
    # RA = RA[:2, :]

    coords_left = cv2.warpAffine(coords_left, LA, (Wm, Hm))
    coords_right = cv2.warpAffine(coords_right, RA, (Wm, Hm))

    coords_left = coords_left - (Wm // 2) + 0.5
    coords_right = coords_right - (Hm // 2) + 0.5

    mesh_label = np.concatenate([coords_left[:,:Wm//2,:], coords_right[:,Wm//2:,:]], axis=1)
    mesh_label = np.moveaxis(mesh_label, -1, 0)

    return mesh_uniform, mesh_label


if __name__ == "__main__":

    wide = cv2.imread('./input/img.png')
    left = cv2.imread('./input/left1.png')
    right = cv2.imread('./input/right1.png')
    seg, box, LH, LA, RH, RA = get_label_masks(wide, left, right)
    mesh_uniform , mesh_label = get_uniform_stereo_mesh(wide, left, right, LH, LA, RH, RA, 4, 46)
    mesh_uniform = mesh_uniform[:, 4:-4, 4:-4].transpose([1, 2, 0])
    mesh_label = mesh_label[:, 4:-4, 4:-4].transpose([1, 2, 0])
    flow = mesh_uniform - mesh_label
    overlay_flow = get_overlay_flow(wide[:, :, ::-1], flow, ratio=0.7)
    overlay_flow = (255 * overlay_flow[:, :, ::-1]).astype(np.uint8)

    plt.figure()
    plt.imshow(overlay_flow)
    plt.show()