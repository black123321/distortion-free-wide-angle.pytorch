import cv2
import numpy as np


def four_vertex_crop(img):
    # mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.sum(img, axis=2)

    # 获取所有非黑像素（四边形区域）
    ys, xs = np.where(mask > 0)
    points = np.stack((xs, ys), axis=1)  # N×2 的点集，每行是 (x, y)

    # 计算四个角点
    sums = points[:, 0] + points[:, 1]  # x + y
    diffs = points[:, 0] - points[:, 1]  # x - y

    left_top = points[np.argmin(sums)]
    right_bottom = points[np.argmax(sums)]
    right_top = points[np.argmax(diffs)]
    left_bottom = points[np.argmin(diffs)]

    pts = np.array([left_top, right_top, right_bottom, left_bottom])

    xs = pts[:, 0]
    ys = pts[:, 1]

    sorted_xs = np.sort(xs)
    sorted_ys = np.sort(ys)

    crop_img = img[sorted_ys[1]:sorted_ys[2]+1, sorted_xs[1]:sorted_xs[2]+1]

    return crop_img