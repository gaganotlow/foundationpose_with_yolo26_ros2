#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoundationPose ROS2 服务节点

提供 /fp_detect 服务（fp_interfaces/FpDetect）：
  - 调用 /yolo_detect 服务获取当前帧 RGB + Depth + Mask
  - 运行 FoundationPose register，返回 6D 位姿（ob_in_cam）

运行要求：
  - 使用 foundationpose_ga conda 环境（含 FoundationPose 全部依赖 + rclpy）
  - 使用 --symlink-install 构建，或设置 FOUNDATIONPOSE_DIR 环境变量

典型用法：
  ros2 launch foundationpose service.launch.py mesh_file:=/path/to/K2.obj class_name:=k2c
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import cv2
import imageio
import numpy as np

# -----------------------------------------------------------------------
# 将 FoundationPose 源码目录加入 Python 路径
# 构建方式为 --symlink-install 时，__file__ 指向源码位置，相对路径即可。
# 其他情况需设置环境变量 FOUNDATIONPOSE_DIR。
# -----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_FP_SRC = os.path.realpath(os.path.join(_THIS_DIR, ".."))

if os.path.exists(os.path.join(_FP_SRC, "estimater.py")):
    if _FP_SRC not in sys.path:
        sys.path.insert(0, _FP_SRC)
else:
    _fp_env = os.environ.get("FOUNDATIONPOSE_DIR", "")
    if _fp_env and os.path.exists(os.path.join(_fp_env, "estimater.py")):
        if _fp_env not in sys.path:
            sys.path.insert(0, _fp_env)
    else:
        raise ImportError(
            "找不到 FoundationPose 源码目录。\n"
            "请使用 --symlink-install 构建，或设置环境变量 FOUNDATIONPOSE_DIR=<FoundationPose 源码路径>"
        )

# FoundationPose 核心模块（需 conda foundationpose_ga 环境）
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped

from yolo_interfaces.srv import YoloDetect
from fp_interfaces.srv import FpDetect


class FpServiceNode(Node):

    def __init__(self):
        super().__init__("foundationpose_service_node")

        self.declare_parameter("mesh_file", "")
        self.declare_parameter("yolo_service", "yolo_detect")
        self.declare_parameter("camera_info_topic", "/right_camera/color/camera_info")
        self.declare_parameter("est_refine_iter", 5)
        self.declare_parameter("debug", 0)
        self.declare_parameter("debug_dir", "/media/rykj/nvme/jetson/ga/code/ros2_ws/src/FoundationPose/fp_debug")
        self.declare_parameter("service_name", "fp_detect")

        self._mesh_file: str = self.get_parameter("mesh_file").value
        self._yolo_svc: str = self.get_parameter("yolo_service").value
        self._ci_topic: str = self.get_parameter("camera_info_topic").value
        self._est_iter: int = self.get_parameter("est_refine_iter").value
        self._debug: int = self.get_parameter("debug").value
        self._debug_dir: str = self.get_parameter("debug_dir").value
        svc_name: str = self.get_parameter("service_name").value

        self._bridge = CvBridge()
        self._K: np.ndarray | None = None
        self._est: FoundationPose | None = None
        self._glctx = None
        self._bbox: np.ndarray | None = None  # (2,3) AABB for visualization
        self._frame_count: int = 0

        # ReentrantCallbackGroup 允许在 service 回调中等待 client future
        cb = ReentrantCallbackGroup()

        self.create_subscription(CameraInfo, self._ci_topic, self._on_ci, 1)

        self._yolo_cli = self.create_client(
            YoloDetect, self._yolo_svc, callback_group=cb
        )
        self._fp_srv = self.create_service(
            FpDetect, svc_name, self._fp_callback, callback_group=cb
        )

        # 提前初始化估计器（如果已配置 mesh_file）
        if self._mesh_file:
            try:
                self._init_estimator(self._mesh_file)
            except Exception as e:
                self.get_logger().error(f"估计器预初始化失败: {e}")

        self.get_logger().info("=" * 60)
        self.get_logger().info("FoundationPose 服务节点已启动")
        self.get_logger().info(f"  服务名称:       /{svc_name}")
        self.get_logger().info(f"  YOLO 服务:      /{self._yolo_svc}")
        self.get_logger().info(f"  相机内参话题:   {self._ci_topic}")
        self.get_logger().info(f"  Mesh 文件:      {self._mesh_file or '(未设置，请在请求前配置 mesh_file 参数)'}")
        self.get_logger().info(f"  精化迭代次数:   {self._est_iter}")
        self.get_logger().info("=" * 60)

    # ------------------------------------------------------------------
    # 相机内参回调
    # ------------------------------------------------------------------
    def _on_ci(self, msg: CameraInfo) -> None:
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f"获取到相机内参 K: fx={self._K[0,0]:.2f} fy={self._K[1,1]:.2f} "
                f"cx={self._K[0,2]:.2f} cy={self._K[1,2]:.2f}"
            )

    # ------------------------------------------------------------------
    # 初始化（或重新加载）FoundationPose 估计器
    # ------------------------------------------------------------------
    def _init_estimator(self, mesh_file: str) -> None:
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh 文件不存在: {mesh_file}")

        self.get_logger().info(f"初始化 FoundationPose 估计器，mesh: {mesh_file}")
        set_seed(0)
        os.makedirs(self._debug_dir, exist_ok=True)
        os.makedirs(os.path.join(self._debug_dir, "vis"), exist_ok=True)

        mesh = trimesh.load(mesh_file)
        extents = mesh.extents
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        if self._glctx is None:
            self._glctx = dr.RasterizeCudaContext()

        self._est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=self._debug_dir,
            debug=self._debug,
            glctx=self._glctx,
        )
        self._mesh_file = mesh_file
        self.get_logger().info("FoundationPose 估计器初始化完成")

    # ------------------------------------------------------------------
    # 服务回调
    # ------------------------------------------------------------------
    def _fp_callback(self, request: FpDetect.Request, response: FpDetect.Response):
        class_name = request.class_name.strip()

        if not class_name:
            response.success = False
            response.message = "class_name 不能为空"
            return response

        if self._est is None:
            response.success = False
            response.message = (
                "估计器未初始化，请确认 mesh_file 参数已正确设置，"
                f"例如: {_FP_SRC}/demo_data/k2c/mesh/03.obj"
            )
            return response

        if self._K is None:
            response.success = False
            response.message = f"尚未收到相机内参，请确认话题 {self._ci_topic} 正常"
            return response

        # ------ 1. 等待 YOLO 服务可用 ------
        if not self._yolo_cli.wait_for_service(timeout_sec=5.0):
            response.success = False
            response.message = f'YOLO 服务 "/{self._yolo_svc}" 不可用，请先启动 YOLO 节点'
            self.get_logger().warning(response.message)
            return response

        # ------ 2. 调用 YOLO 服务 ------
        yolo_req = YoloDetect.Request()
        yolo_req.class_name = class_name
        future = self._yolo_cli.call_async(yolo_req)

        deadline = time.monotonic() + 30.0
        while not future.done():
            if time.monotonic() > deadline:
                response.success = False
                response.message = "YOLO 服务调用超时（30s）"
                return response
            time.sleep(0.01)

        yolo_resp = future.result()
        if yolo_resp is None or not yolo_resp.success or not yolo_resp.detections:
            msg = getattr(yolo_resp, "message", "无响应") if yolo_resp else "无响应"
            response.success = False
            response.message = f"YOLO 未检测到目标: {msg}"
            self.get_logger().warning(response.message)
            return response

        # ------ 3. 解码图像 ------
        try:
            best = max(yolo_resp.detections, key=lambda d: float(d.confidence))
            rgb = self._decode_rgb(yolo_resp.rgb)
            depth = self._decode_depth(yolo_resp.depth)
            mask = self._decode_mask(best.mask)
        except Exception as e:
            response.success = False
            response.message = f"图像解码失败: {e}"
            self.get_logger().error(traceback.format_exc())
            return response

        if rgb.shape[:2] != depth.shape[:2]:
            response.success = False
            response.message = (
                f"RGB 与 depth 尺寸不一致: rgb={rgb.shape}, depth={depth.shape}"
            )
            return response

        # ------ 4. FoundationPose register ------
        try:
            iter_count = (
                request.est_refine_iter if request.est_refine_iter > 0 else self._est_iter
            )
            self.get_logger().info(
                f"开始 register: class={class_name}, iter={iter_count}, "
                f"mask_px={int(mask.sum())}"
            )
            pose = self._est.register(
                K=self._K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iter_count
            )
        except Exception as e:
            response.success = False
            response.message = f"FoundationPose register 失败: {e}"
            self.get_logger().error(traceback.format_exc())
            return response

        # ------ 5. 构造 PoseStamped ------
        T = pose.reshape(4, 4)
        t = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (x, y, z, w)

        ps = PoseStamped()
        ps.header = yolo_resp.header
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])

        response.success = True
        response.message = (
            f"class={class_name} "
            f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m"
        )
        response.pose = ps
        self.get_logger().info(f"位姿估计成功: {response.message}")

        # ------ 6. 保存可视化（RGB + 3D 框 + 坐标轴） ------
        self._save_vis(rgb, pose, class_name)

        return response

    def _save_vis(self, rgb: np.ndarray, pose: np.ndarray, class_name: str) -> None:
        """在 RGB 图上绘制 3D 包围框和坐标轴，保存到 debug_dir/vis/。"""
        if self._bbox is None or self._K is None:
            return
        try:
            vis = draw_posed_3d_box(self._K, img=rgb, ob_in_cam=pose, bbox=self._bbox)
            vis = draw_xyz_axis(
                vis, ob_in_cam=pose, scale=0.1, K=self._K,
                thickness=3, transparency=0, is_input_rgb=True,
            )
            self._frame_count += 1
            out_path = os.path.join(
                self._debug_dir, "vis", f"{class_name}_{self._frame_count:06d}.png"
            )
            imageio.imwrite(out_path, vis)
            self.get_logger().info(f"可视化已保存: {out_path}")
        except Exception:
            self.get_logger().warning(f"可视化保存失败:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 图像解码工具
    # ------------------------------------------------------------------
    def _decode_rgb(self, img_msg) -> np.ndarray:
        enc = (img_msg.encoding or "").lower()
        if enc == "rgb8":
            return np.asarray(self._bridge.imgmsg_to_cv2(img_msg, "rgb8"), dtype=np.uint8)
        bgr = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _decode_depth(self, img_msg) -> np.ndarray:
        enc = (img_msg.encoding or "").lower()
        if enc == "32fc1":
            return np.asarray(
                self._bridge.imgmsg_to_cv2(img_msg, "passthrough"), dtype=np.float32
            )
        d = self._bridge.imgmsg_to_cv2(img_msg, "passthrough")
        return d.astype(np.float32) * 0.001  # uint16 毫米 → float32 米

    def _decode_mask(self, img_msg) -> np.ndarray:
        m = self._bridge.imgmsg_to_cv2(img_msg, "mono8")
        return (m > 127).astype(bool)


def main():
    set_logging_format()
    rclpy.init()
    node = FpServiceNode()
    # MultiThreadedExecutor 允许 service 回调在等待 YOLO future 时不阻塞其他回调
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
