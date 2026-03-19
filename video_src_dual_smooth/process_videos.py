import gc
import os, cv2, argparse, tempfile, shutil, sys

from utils import four_vertex_crop

sys.path.insert(0, os.getcwd())
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from data import ImageDataset
from energy import Energy
from visualization import get_overlay_flow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import imageio


# 你已存在的依赖：
# from dataset import ImageDataset
# from energy import Energy
# from vis import get_overlay_flow  # 你上面贴的函数


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mesh(last_mesh, frame_bgr, args, dataset, options, predictor):
    """
       对单帧做一次畸变修复，返回修复后帧、光流可视化(可选)。
       为了兼容现有 ImageDataset.get_image_by_file，这里把帧写到临时文件再加载。
       """
    ori_h, ori_w = frame_bgr.shape[:2]

    # 走原有的数据准备流程
    image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded, seg_mask = dataset.get_image_by_file(
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
    last_mesh = model.mesh.detach()
    mesh_optimal = model.mesh.detach().cpu().numpy()

    return mesh_optimal, last_mesh, seg_mask

def process_one_frame(image, mesh_optimal, resize, save_flow_path, i):
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
    np.save(os.path.join(save_flow_path, f"{i:06d}.npy"), mesh_optimal)

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

def main_video(video_path, out_video_path, out_flow_path, out_mask_path, args):
    forward_mesh_list = []
    backward_mesh_list = []
    assert video_path is not None and os.path.exists(video_path), "请提供有效 --video 路径"

    # 输出目录与视频写入器
    os.makedirs(out_video_path, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(out_video_path, f"{base}.mp4")
    out_flow_path = os.path.join(out_flow_path, f"{base}")
    out_mask_path = os.path.join(out_mask_path, f"{base}")
    os.makedirs(out_flow_path, exist_ok=True)
    os.makedirs(out_mask_path, exist_ok=True)

    frames_dir = os.path.join(args.out_dir, f"{base}_frames") if args.save_frames else None
    flows_dir = os.path.join(args.out_dir, f"{base}_flows") if args.save_flow_overlay else None
    if frames_dir: os.makedirs(frames_dir, exist_ok=True)
    if flows_dir: os.makedirs(flows_dir, exist_ok=True)

    forward_cap = cv2.VideoCapture(video_path)
    backward_cap = cv2.VideoCapture(video_path)
    if not forward_cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

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
            forward_mesh, forward_last_mesh, seg_mask = get_mesh(forward_last_mesh, frame_bgr, args, dataset, options, predictor)
            backward_mesh, backward_last_mesh, _ = get_mesh(backward_last_mesh, backward_frame, args, dataset, options, predictor)
            forward_mesh_list.append(forward_mesh)
            backward_mesh_list.append(backward_mesh)
            seg_mask = seg_mask.astype(np.float32)

            np.save(os.path.join(out_mask_path, f"{idx:06d}.npy"), seg_mask)
            # 写出视频帧
            # writer.append_data(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            idx += 1
            pbar.update(1)
            inverse_idx = inverse_idx-1
        pbar.close()
        backward_mesh_list.reverse()
        backward_mesh_list = np.array(backward_mesh_list)
        forward_mesh_list = np.array(forward_mesh_list)
        mesh_list = (forward_mesh_list + backward_mesh_list) / 2
        cap = cv2.VideoCapture(video_path)
        print('start wrap video')
        idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            corrected = process_one_frame(frame_bgr, mesh_list[idx], args.resize, out_flow_path, idx)
            writer.append_data(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))

            idx += 1
    finally:
        forward_cap.release()
        backward_cap.release()
        writer.close()
        # writer.release()
        del forward_mesh_list
        del backward_mesh_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[Done] 输出视频: {out_video_path}")
    if frames_dir: print(f"[Info] 修复帧保存于: {frames_dir}")
    if flows_dir: print(f"[Info] 光流叠加保存于: {flows_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 原有参数
    parser = argparse.ArgumentParser(description='Distortion-Free-Wide-Angle-Portraits-on-Camera-Phones')
    parser.add_argument('--num_iter', type=int, default=200, help="number of optimization steps") # 1k-200; 4k-300
    parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
    parser.add_argument('--Q', type=int, default=18, help="number of padding vertices")
    parser.add_argument('--mesh_ds_ratio', type=int, default=46, help="the pixel-to-vertex ratio") # 1k-24; 4k-46

    parser.add_argument('--naive', type=int, default=0, help="if set True, perform naive orthographic correction")
    parser.add_argument('--face_energy', type=float, default=4, help="weight of the face energy term")
    parser.add_argument('--similarity', type=int, default=1, help="weight of similarity tranformation constraint")
    parser.add_argument('--line_bending', type=float, default=10, help="weight of the line bending term")
    parser.add_argument('--regularization', type=float, default=0.5, help="weight of the regularization term")
    parser.add_argument('--boundary_constraint', type=float, default=4, help="weight of the mesh boundary constraint")
    parser.add_argument('--time_energy', type=float, default=25, help="weight of the mesh boundary constraint")

    # 新增视频参数
    parser.add_argument("--video", type=str, default='/media/ubuntu/1410bddb-88a9-4324-a212-78a014d836dc/Datasets/wide_range_video/video/pura80 pro/clip_processed/wide', help="输入视频路径")
    parser.add_argument("--out_dir", type=str, default="./smooth_correction(H264 timeE25 padding)", help="输出目录")
    parser.add_argument("--save_frames", action="store_true", help="是否保存修复后的每帧图片")
    parser.add_argument("--save_flow_overlay", action="store_true", help="是否保存光流叠加图")
    parser.add_argument("--resize", type=int, default=-1)

    args = parser.parse_args()

    videos_list = []
    videos_path = args.video
    for scene in os.listdir(videos_path):
        video_path = os.path.join(videos_path, scene)
        for video_name in os.listdir(video_path):
            videos_list.append(os.path.join(video_path, video_name))
        videos_list.sort()

    for idx, video_path in enumerate(videos_list):
        print(f'正在处理第{idx+1}/{len(videos_list)}个视频: {video_path}')
        out_video_path = args.out_dir + "/" + video_path.split("/")[-2]
        out_flow_path = '../smooth_correction_flow(timeE25 padding)' + "/" + video_path.split("/")[-2]
        out_mask_path = '../smooth_correction_mask(timeE25 padding)' + "/" + video_path.split("/")[-2]
        main_video(video_path, out_video_path, out_flow_path, out_mask_path, args)

