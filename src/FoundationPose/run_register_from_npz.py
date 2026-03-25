#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 ros_dump_yolo_for_foundationpose.py 保存的 .npz，用原版 FoundationPose 做一次 register（6D）。

依赖：本仓库 conda 环境（见 readme.md），与 Isaac ROS 无关。

一键串联（推荐）：
  bash /media/rykj/nvme/jetson/ga/code/FoundationPose/run_yolo_fp_original.sh --class-name k2c

仅跑第二步时：
  cd /media/rykj/nvme/jetson/ga/code/FoundationPose
  conda activate foundationpose
  python run_register_from_npz.py \\
    --mesh /media/rykj/nvme/jetson/K2.obj \\
    --npz /tmp/fp_input.npz \\
    --pose-out /tmp/ob_in_cam.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import trimesh

# 与 run_demo_k2c.py 一致：在仓库根目录执行
_CODE_DIR = os.path.dirname(os.path.realpath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import set_logging_format, set_seed
import nvdiffrast.torch as dr


def print_pose_6d_human(pose44: np.ndarray) -> None:
    T = np.asarray(pose44).reshape(4, 4)
    t = T[:3, 3]
    R = T[:3, :3]
    print("\n" + "=" * 56)
    print(" 6D 位姿（物体相对相机坐标系 ob_in_cam）")
    print("=" * 56)
    print(f" 平移 (m):  tx={t[0]:+.6f}  ty={t[1]:+.6f}  tz={t[2]:+.6f}")
    try:
        from scipy.spatial.transform import Rotation as SciR

        q = SciR.from_matrix(R).as_quat()
        print(
            f" 四元数 (x,y,z,w):  {q[0]:+.6f} {q[1]:+.6f} {q[2]:+.6f} {q[3]:+.6f}"
        )
        euler = SciR.from_matrix(R).as_euler("xyz", degrees=True)
        print(
            f" 欧拉角 (deg, xyz):  roll={euler[0]:+.2f}  pitch={euler[1]:+.2f}  yaw={euler[2]:+.2f}"
        )
    except Exception:
        print(" 旋转矩阵 R (3x3):")
        print(R)
    print(" 齐次矩阵 4x4:")
    for row in T:
        print("  ", " ".join(f"{x:+.6f}" for x in row))
    print("=" * 56 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True, help="物体 mesh，如 K2.obj")
    parser.add_argument("--npz", type=str, required=True, help="rgb/depth/ob_mask/K 的 npz")
    parser.add_argument("--pose-out", type=str, default="/tmp/ob_in_cam.txt", help="4x4 位姿文本")
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug-dir", type=str, default=f"{_CODE_DIR}/debug_ros_npz")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    print(
        "\n>>> 正在计算 6D 位姿（加载 FoundationPose 预训练模型 + register）...\n",
        flush=True,
    )

    data = np.load(args.npz, allow_pickle=True)
    rgb = data["rgb"]
    depth = data["depth"]
    ob_mask = data["ob_mask"].astype(bool)
    K = data["K"].reshape(3, 3)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    mesh = trimesh.load(args.mesh)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("estimator init done, running register ...")

    pose = est.register(
        K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=args.est_refine_iter
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.pose_out)) or ".", exist_ok=True)
    np.savetxt(args.pose_out, pose.reshape(4, 4))
    logging.info("pose (4x4, ob_in_cam) saved to %s", args.pose_out)
    print_pose_6d_human(pose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
