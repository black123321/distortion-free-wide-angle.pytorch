import os, cv2, argparse, tempfile, shutil, sys

from utils import four_vertex_crop

sys.path.insert(0, os.getcwd())
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from video_src.data import ImageDataset
from video_src.energy import Energy
from video_src.visualization import get_overlay_flow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


# 你已存在的依赖：
# from dataset import ImageDataset
# from energy import Energy
# from vis import get_overlay_flow  # 你上面贴的函数
def process_one_frame(last_mesh, frame_bgr, args, dataset, options, predictor):
    """
    对单帧做一次畸变修复，返回修复后帧、光流可视化(可选)。
    为了兼容现有 ImageDataset.get_image_by_file，这里把帧写到临时文件再加载。
    """
    ori_h, ori_w = frame_bgr.shape[:2]
    # 暂存到临时文件（.png 无损，减少反复 jpeg 失真）
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = tf.name
    cv2.imwrite(tmp_path, frame_bgr)

    # 走原有的数据准备流程
    image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded = dataset.get_image_by_file(
        tmp_path, resize=args.resize, predictor=predictor)
    H, W = image.shape[:2]
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
                   correction_strength, box_masks_padded, seg_mask_padded, args.Q).to("cpu")
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
    mesh_optimal = mesh_optimal[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    flow = mesh_uniform - mesh_optimal  # (h_m, w_m, 2)

    map_optimal = cv2.resize(mesh_optimal, (W, H))
    # remap 需要 float32 且是“源图坐标”
    x_map = (map_optimal[:, :, 0] + W // 2).astype(np.float32)
    y_map = (map_optimal[:, :, 1] + H // 2).astype(np.float32)
    out = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101)
    # out = four_vertex_crop(out)
    out = cv2.resize(out, (ori_w, ori_h), interpolation=cv2.INTER_AREA)

    # 光流叠加图（可选）
    overlay = None
    if getattr(args, "save_flow_overlay", False):
        # get_overlay_flow 期待 RGB
        overlay_flow = get_overlay_flow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), flow, ratio=0.7)
        overlay = (overlay_flow[:, :, ::-1] * 255.0).astype(np.uint8)  # 回到 BGR uint8

    # 清理临时文件
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return out, overlay, last_mesh

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
    assert args.video is not None and os.path.exists(args.video), "请提供有效 --video 路径"

    # 输出目录与视频写入器
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_video_path = os.path.join(args.out_dir, f"{base}_corrected_smooth.mp4")
    frames_dir = os.path.join(args.out_dir, f"{base}_frames") if args.save_frames else None
    flows_dir = os.path.join(args.out_dir, f"{base}_flows") if args.save_flow_overlay else None
    if frames_dir: os.makedirs(frames_dir, exist_ok=True)
    if flows_dir: os.makedirs(flows_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

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
    last_mesh = None
    idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Processing")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # 保证输入给 remap 的尺寸与视频一致
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            corrected, overlay, last_mesh = process_one_frame(last_mesh, frame_bgr, args, dataset, options, predictor)

            # 写出视频帧
            writer.write(corrected)

            # 可选：保存帧与光流叠加
            if frames_dir:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:06d}.jpg"), corrected, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if flows_dir and overlay is not None:
                cv2.imwrite(os.path.join(flows_dir, f"flow_{idx:06d}.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"[Done] 输出视频: {out_video_path}")
    if frames_dir: print(f"[Info] 修复帧保存于: {frames_dir}")
    if flows_dir: print(f"[Info] 光流叠加保存于: {flows_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 原有参数
    parser = argparse.ArgumentParser(description='Distortion-Free-Wide-Angle-Portraits-on-Camera-Phones')
    parser.add_argument('--num_iter', type=int, default=200, help="number of optimization steps") # 1k-200; 4k-300
    parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
    parser.add_argument('--Q', type=int, default=4, help="number of padding vertices")
    parser.add_argument('--mesh_ds_ratio', type=int, default=40, help="the pixel-to-vertex ratio") # 1k-24; 4k-46

    parser.add_argument('--naive', type=int, default=0, help="if set True, perform naive orthographic correction")
    parser.add_argument('--face_energy', type=float, default=4, help="weight of the face energy term")
    parser.add_argument('--similarity', type=int, default=1, help="weight of similarity tranformation constraint")
    parser.add_argument('--line_bending', type=float, default=4, help="weight of the line bending term")
    parser.add_argument('--regularization', type=float, default=0.5, help="weight of the regularization term")
    parser.add_argument('--boundary_constraint', type=float, default=4, help="weight of the mesh boundary constraint")
    parser.add_argument('--time_energy', type=float, default=4, help="weight of the mesh boundary constraint")

    # 新增视频参数
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--out_dir", type=str, default="results_video", help="输出目录")
    parser.add_argument("--save_frames", action="store_true", help="是否保存修复后的每帧图片")
    parser.add_argument("--save_flow_overlay", action="store_true", help="是否保存光流叠加图")
    parser.add_argument("--resize", type=int, default=-1)

    args = parser.parse_args()

    main_video(args)

