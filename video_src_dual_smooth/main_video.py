import os, cv2, argparse, tempfile, shutil, sys

from utils import four_vertex_crop

sys.path.insert(0, os.getcwd())
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from video_src_dual_smooth.data import ImageDataset
from video_src_dual_smooth.energy import Energy
from video_src_dual_smooth.visualization import get_overlay_flow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import imageio


# 你已存在的依赖：
# from dataset import ImageDataset
# from energy import Energy
# from vis import get_overlay_flow  # 你上面贴的函数

forward_mesh_list = []
backward_mesh_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mesh(last_mesh, frame_bgr, args, dataset, options, predictor):
    """
       对单帧做一次畸变修复，返回修复后帧、光流可视化(可选)。
       为了兼容现有 ImageDataset.get_image_by_file，这里把帧写到临时文件再加载。
       """
    # 走原有的数据准备流程
    image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded, _ = dataset.get_image_by_file(
        frame_bgr, resize=args.resize, predictor=predictor)
    # 组装 energy 选项
    if args.naive:
        trivial_mask = np.ones_like(correction_strength)
        box_masks_padded = trivial_mask[np.newaxis, :, :]
        seg_mask_padded = trivial_mask
        local_options = {
            "face_energy": 4,
            "similarity": False,
            "line_bending": 0,
            "regularization": 0,
            "boundary_constraint": 0
        }
    else:
        local_options = options  # 直接复用传入

    # 构建/优化
    model = Energy(local_options, mesh_uniform_padded, mesh_stereo_padded, last_mesh,
                   correction_strength, box_masks_padded, seg_mask_padded, args.Q).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for _ in range(args.num_iter):
        optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        optimizer.step()

    # 计算光流并重采样
    mesh_uniform = mesh_uniform_padded[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    last_mesh = model.mesh
    mesh_optimal = model.mesh.detach().cpu().numpy()

    return mesh_optimal, last_mesh

def process_one_frame(image, mesh_optimal, resize):
    ori_h, ori_w, _ = image.shape
    if resize > 0:
        min_side = min(ori_h, ori_w)
        new_h = int(ori_h / min_side * resize)
        new_w = int(ori_w / min_side * resize)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_size = 80
    image = np.pad(image, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], "constant", constant_values=0)
    H, W, _ = image.shape
    # 计算光流并重采样
    mesh_optimal = mesh_optimal[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])

    map_optimal = cv2.resize(mesh_optimal, (W, H))
    # remap 需要 float32 且是“源图坐标”
    x_map = (map_optimal[:, :, 0] + W // 2).astype(np.float32)
    y_map = (map_optimal[:, :, 1] + H // 2).astype(np.float32)
    out = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101)
    pad_size = 80
    out = out[pad_size:-pad_size, pad_size:-pad_size]
    out = cv2.resize(out, (ori_w, ori_h), interpolation=cv2.INTER_AREA)

    return out

def build_predictor(cfg_name="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    predictor = DefaultPredictor(cfg)
    return predictor

def main_video(args):
    global backward_mesh_list, forward_mesh_list
    assert args.video is not None and os.path.exists(args.video), "请提供有效 --video 路径"

    # 输出目录与视频写入器
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_video_path = os.path.join(args.out_dir, f"{base}_corrected_smooth{args.time_energy}(padding).mp4")
    frames_dir = os.path.join(args.out_dir, f"{base}_frames") if args.save_frames else None
    flows_dir = os.path.join(args.out_dir, f"{base}_flows") if args.save_flow_overlay else None
    if frames_dir: os.makedirs(frames_dir, exist_ok=True)
    if flows_dir: os.makedirs(flows_dir, exist_ok=True)

    forward_cap = cv2.VideoCapture(args.video)
    backward_cap = cv2.VideoCapture(args.video)
    if not forward_cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = forward_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(forward_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(forward_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    writer = imageio.get_writer(out_video_path, fps=fps, codec="libx264", quality=8)

    # 数据集与共享 options
    dataset = ImageDataset(args)
    options = {
        "face_energy": args.face_energy,
        "similarity": args.similarity,
        "line_bending": args.line_bending,
        "regularization": args.regularization,
        "boundary_constraint": args.boundary_constraint,
        "time_energy": args.time_energy
    }

    predictor = build_predictor()
    forward_last_mesh = None
    backward_last_mesh = None
    idx = 0
    frame_count = int(forward_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inverse_idx = frame_count-1
    pbar = tqdm(total=int(forward_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Processing")
    try:
        while True:
            ret, frame_bgr = forward_cap.read()
            if not ret:
                break
            backward_cap.set(cv2.CAP_PROP_POS_FRAMES, inverse_idx)
            _, backward_frame = backward_cap.read()
            forward_mesh, forward_last_mesh = get_mesh(forward_last_mesh, frame_bgr, args, dataset, options, predictor)
            backward_mesh, backward_last_mesh = get_mesh(backward_last_mesh, backward_frame, args, dataset, options, predictor)
            forward_mesh_list.append(forward_mesh)
            backward_mesh_list.append(backward_mesh)

            # 写出视频帧
            # writer.append_data(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))

            pbar.update(1)
            inverse_idx = inverse_idx-1
        backward_mesh_list.reverse()
        backward_mesh_list = np.array(backward_mesh_list)
        forward_mesh_list = np.array(forward_mesh_list)
        mesh_list = (forward_mesh_list + backward_mesh_list) / 2
        cap = cv2.VideoCapture(args.video)
        print('start wrap video')
        idx = 0
        pbar = tqdm(total=int(len(mesh_list)), desc="Processing")
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            corrected = process_one_frame(frame_bgr, mesh_list[idx], args.resize)
            writer.append_data(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))

            idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.close()
        # writer.release()

    print(f"[Done] 输出视频: {out_video_path}")
    if frames_dir: print(f"[Info] 修复帧保存于: {frames_dir}")
    if flows_dir: print(f"[Info] 光流叠加保存于: {flows_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 原有参数
    parser = argparse.ArgumentParser(description='Distortion-Free-Wide-Angle-Portraits-on-Camera-Phones')
    parser.add_argument('--num_iter', type=int, default=200, help="number of optimization steps") # 1k-200; 4k-300
    parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
    parser.add_argument('--Q', type=int, default=12, help="number of padding vertices")
    parser.add_argument('--mesh_ds_ratio', type=int, default=46, help="the pixel-to-vertex ratio") # 1k-24; 4k-46

    parser.add_argument('--naive', type=int, default=0, help="if set True, perform naive orthographic correction")
    parser.add_argument('--face_energy', type=float, default=4, help="weight of the face energy term")
    parser.add_argument('--similarity', type=int, default=1, help="weight of similarity tranformation constraint")
    parser.add_argument('--line_bending', type=float, default=10, help="weight of the line bending term")
    parser.add_argument('--regularization', type=float, default=0.5, help="weight of the regularization term")
    parser.add_argument('--boundary_constraint', type=float, default=4, help="weight of the mesh boundary constraint")
    parser.add_argument('--time_energy', type=float, default=25, help="weight of the mesh boundary constraint")

    # 新增视频参数
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--out_dir", type=str, default="./video_src_dual_smooth/results_video", help="输出目录")
    parser.add_argument("--save_frames", action="store_true", help="是否保存修复后的每帧图片")
    parser.add_argument("--save_flow_overlay", action="store_true", help="是否保存光流叠加图")
    parser.add_argument("--resize", type=int, default=-1)

    args = parser.parse_args()

    main_video(args)

