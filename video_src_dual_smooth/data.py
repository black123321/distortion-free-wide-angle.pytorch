import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from stereographic import get_uniform_stereo_mesh
from perception import get_face_masks, get_object_masks


class ImageDataset():

    def __init__(self, args, root='data'):

        self.Q = args.Q
        self.mesh_ds_ratio = args.mesh_ds_ratio
        # self.data_list = []
        # for names in os.listdir(root):
        #     if names.endswith(".jpg"):
        #         self.data_list.append(os.path.join(root, names))
        # self.data_list = sorted(self.data_list)


    def get_image_by_file(self, file, resize=-1, classes=None, predictor=None):
        data_name = file
        # try:
        #     fov = int(data_name.split('/')[-1].split('.')[0].split('_')[-1])
        # except ValueError:
        fov = 100
        image = cv2.imread(data_name)
        H, W, _ = image.shape
        if resize > 0:
            min_side = min(H, W)
            new_h = int(H / min_side * resize)
            new_w = int(W / min_side * resize)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_size = 40
        image = np.pad(image, [[pad_size, pad_size], [pad_size, pad_size],[0, 0]], "constant", constant_values=0)
        H, W, _ = image.shape
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        Hm = H // self.mesh_ds_ratio
        Wm = W // self.mesh_ds_ratio

        if classes is None:
            seg_mask, box_masks = get_face_masks(image, predictor=predictor)
        else:
            seg_mask, box_masks = get_object_masks(image, classes=classes, predictor=predictor)

        seg_mask = cv2.resize(seg_mask.astype(np.float32), (Wm, Hm))
        box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm, Hm)) for box_mask in box_masks]
        # print(box_masks)
        box_masks = np.stack(box_masks, axis=0)
        seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
        box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")
        mesh_uniform_padded, mesh_stereo_padded = get_uniform_stereo_mesh(image, fov * np.pi / 180, self.Q, self.mesh_ds_ratio)
        radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
        half_diagonal = np.linalg.norm([H + 2 * self.Q * self.mesh_ds_ratio, W + 2 * self.Q * self.mesh_ds_ratio]) / 2.
        ra = half_diagonal / 1.8
        rb = half_diagonal / (3 * np.log(99))
        correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))

        return image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded

    def preprocess_image(self, file, resize=-1, classes=None, predictor=None):
        data_name = file
        try:
            fov = int(data_name.split('/')[-1].split('.')[0].split('_')[-1])
        except ValueError:
            fov = 120
        image = cv2.imread(data_name)
        H, W, _ = image.shape
        if resize > 0:
            min_side = min(H, W)
            new_h = int(H / min_side * resize)
            new_w = int(W / min_side * resize)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_size = 80
        image = np.pad(image, [[pad_size, pad_size], [pad_size, pad_size],[0, 0]], "constant", constant_values=0)
        H, W, _ = image.shape
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        Hm = H // self.mesh_ds_ratio
        Wm = W // self.mesh_ds_ratio

        if classes is None:
            seg_mask, box_masks = get_face_masks(image, predictor=predictor)
        else:
            seg_mask, box_masks = get_object_masks(image, classes=classes, predictor=predictor)

        seg_mask = cv2.resize(seg_mask.astype(np.float32), (Wm, Hm))
        box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm, Hm)) for box_mask in box_masks]
        box_masks = np.stack(box_masks, axis=0)
        seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
        box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")
        mesh_uniform_padded, mesh_stereo_padded = get_uniform_stereo_mesh(image, fov * np.pi / 180, self.Q, self.mesh_ds_ratio)
        radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
        half_diagonal = np.linalg.norm([H + 2 * self.Q * self.mesh_ds_ratio, W + 2 * self.Q * self.mesh_ds_ratio]) / 2.
        ra = half_diagonal / 2
        rb = half_diagonal / (2.5 * np.log(99))
        correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))

        return image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded

