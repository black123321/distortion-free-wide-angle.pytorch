import cv2
import os
import numpy as np
from torch.utils.data import Dataset

from stereographic import get_uniform_stereo_mesh
from perception import get_face_masks, get_label_masks



class ImageDataset(Dataset):

    def __init__(self, args, root='data'):

        self.Q = args.Q
        self.mesh_ds_ratio = args.mesh_ds_ratio
        self.data_list = []
        for names in os.listdir(root):
            if names.endswith(".jpg"):
                self.data_list.append(os.path.join(root, names))
        self.data_list = sorted(self.data_list)


    def get_image_by_file(self, file, left_label, right_label):
        data_name = file

        image = cv2.imread(data_name)
        left_label = cv2.imread(left_label)
        right_label = cv2.imread(right_label)
        h, w = image.shape[:2]
        lh, lw = left_label.shape[:2]
        left_label = cv2.resize(left_label, (int(lw * h / lh), h))
        right_label = cv2.resize(right_label, (int(lw * h / lh), h))


        H, W, _ = image.shape

        Hm = H // self.mesh_ds_ratio
        Wm = W // self.mesh_ds_ratio

        seg_mask, box_masks = get_label_masks(image, left_label, right_label)

        seg_mask = cv2.resize(seg_mask.astype(np.float32), (Wm, Hm))
        box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm, Hm)) for box_mask in box_masks]
        # print(box_masks)
        box_masks = np.stack(box_masks, axis=0)
        seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
        box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")
        mesh_uniform_padded, mesh_stereo_padded = get_uniform_stereo_mesh(image, left_label, right_label, self.Q, self.mesh_ds_ratio)
        radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
        half_diagonal = np.linalg.norm([H + 2 * self.Q * self.mesh_ds_ratio, W + 2 * self.Q * self.mesh_ds_ratio]) / 2.
        ra = half_diagonal / 2.
        rb = half_diagonal / (2 * np.log(99))
        correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))

        return image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded


    def __getitem__(self, index):

        index = index % len(self.data_list)
        data_name = self.data_list[index]

        return self.get_image_by_file(data_name)


    def __len__(self):
        return len(self.data_list)