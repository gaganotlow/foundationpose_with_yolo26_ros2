#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
从 ROS2 拉 YOLO 服务结果 + 相机内参，保存为 FoundationPose 原版可用的 .npz。

典型用法：先在本终端运行本脚本，再在另一终端启动 YOLO 节点；脚本会等待服务就绪并
在「当前帧未检测到目标」时自动重试，直到拿到 rgb/depth/mask。

数据流：
  - K：来自 CameraInfo 话题。
  - RGB / Depth / Mask：仅在 yolo_detect 成功返回且含检测时才有。

用法（须用系统 Python 3.10 + 已 source ros2_ws）：
  python3 ros_dump_yolo_for_foundationpose.py --class-name k2c --out /tmp/fp_input.npz

或一键：bash run_yolo_fp_original.sh --class-name k2c
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

from yolo_interfaces.srv import YoloDetect


def k_from_camera_info(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def depth_to_meters_float(depth_msg, bridge: CvBridge) -> np.ndarray:
    enc = (depth_msg.encoding or "").lower()
    if enc == "32fc1":
        d = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        return np.asarray(d, dtype=np.float32)
    if enc in ("16uc1", "mono16"):
        d = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        return d.astype(np.float32) * 0.001
    d = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    return np.asarray(d, dtype=np.float32)


def rgb_to_uint8_rgb(rgb_msg, bridge: CvBridge) -> np.ndarray:
    enc = (rgb_msg.encoding or "").lower()
    if enc == "rgb8":
        return np.asarray(bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8"))
    if enc == "bgr8":
        import cv2
        bgr = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
    return np.asarray(img)[..., :3]


def mask_to_bool(mask_msg, bridge: CvBridge) -> np.ndarray:
    m = bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
    return (m > 127).astype(bool)


class DumpNode(Node):
    def __init__(self, camera_info_topic: str):
        super().__init__("fp_ros_dump_yolo")
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._ci_topic = camera_info_topic
        self.create_subscription(CameraInfo, self._ci_topic, self._on_ci, 10)

    def _on_ci(self, msg: CameraInfo):
        self._camera_info = msg

    def wait_camera_info(self, timeout_sec: float) -> CameraInfo:
        t0 = time.monotonic()
        last_print = t0
        while time.monotonic() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._camera_info is not None:
                return self._camera_info
            now = time.monotonic()
            if now - last_print >= 10.0:
                self.get_logger().warn(
                    f"仍在等待 CameraInfo: {self._ci_topic} （已等待 {int(now - t0)}s）"
                )
                last_print = now
        raise RuntimeError(f"未收到 CameraInfo: {self._ci_topic}")


def wait_yolo_service(node: Node, cli, service_name: str, timeout_sec: float) -> None:
    t0 = time.monotonic()
    last_log = 0.0
    while time.monotonic() - t0 < timeout_sec:
        if cli.wait_for_service(timeout_sec=1.0):
            node.get_logger().info(f"YOLO 服务已就绪: {service_name}")
            return
        if time.monotonic() - t0 - last_log > 8.0:
            print(
                f"\n[提示] 可在另一终端启动 YOLO 节点；仍在等待服务 '{service_name}' "
                f"（已 {int(time.monotonic() - t0)}s / {int(timeout_sec)}s）\n",
                file=sys.stderr,
                flush=True,
            )
            last_log = time.monotonic() - t0
    raise RuntimeError(
        f"超时：在 {timeout_sec}s 内未等到 YOLO 服务 '{service_name}'"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", required=True, help="传给 yolo_detect 的类别，如 k2c / j2")
    ap.add_argument("--yolo-service", default="yolo_detect")
    ap.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    ap.add_argument("--out", default="/tmp/fp_input.npz", help="输出 .npz 路径")
    ap.add_argument("--ci-timeout", type=float, default=120.0, help="等待 CameraInfo 秒数")
    ap.add_argument("--yolo-call-timeout", type=float, default=90.0, help="单次 yolo_detect 调用超时")
    ap.add_argument(
        "--wait-yolo-service-sec",
        type=float,
        default=600.0,
        help="等待 /yolo_detect 服务出现的最长时间（秒），便于另一终端后启动 YOLO",
    )
    ap.add_argument(
        "--detect-retry-interval",
        type=float,
        default=2.0,
        help="未检测到目标时，间隔多少秒再请求一次",
    )
    ap.add_argument(
        "--detect-retry-for-sec",
        type=float,
        default=0.0,
        help="检测重试最长多久（0=不限制，直到成功）",
    )
    args, ros_argv = ap.parse_known_args()

    print(
        "\n>>> 本步骤将等待：① CameraInfo  ② YOLO 服务 ③ 当前帧成功检出目标。\n"
        ">>> 若尚未启动 YOLO，请在另一终端启动后再观察本窗口（会自动重试）。\n",
        file=sys.stderr,
        flush=True,
    )

    rclpy.init(args=ros_argv)
    node = DumpNode(args.camera_info_topic)
    try:
        node.get_logger().info(f"等待 CameraInfo: {args.camera_info_topic}")
        ci = node.wait_camera_info(args.ci_timeout)
        K = k_from_camera_info(ci)

        cli = node.create_client(YoloDetect, args.yolo_service)
        wait_yolo_service(node, cli, args.yolo_service, args.wait_yolo_service_sec)

        t_retry0 = time.monotonic()
        attempt = 0
        resp = None
        while True:
            attempt += 1
            if args.detect_retry_for_sec > 0:
                if time.monotonic() - t_retry0 > args.detect_retry_for_sec:
                    raise RuntimeError(
                        f"在 {args.detect_retry_for_sec}s 内重试仍未检测到目标，已放弃"
                    )

            req = YoloDetect.Request()
            req.class_name = args.class_name
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(
                node, fut, timeout_sec=args.yolo_call_timeout
            )
            resp = fut.result()
            if resp is None:
                node.get_logger().warn("YOLO 调用无响应，重试 ...")
                time.sleep(args.detect_retry_interval)
                continue
            if resp.success and resp.detections:
                node.get_logger().info(
                    f"已收到 YOLO 数据（第 {attempt} 次请求），共 {len(resp.detections)} 个实例"
                )
                break
            msg = resp.message or ""
            print(
                f"\n[等待检测] {msg}  （{args.detect_retry_interval}s 后重试，第 {attempt} 次）\n",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(args.detect_retry_interval)

        best = max(resp.detections, key=lambda d: float(d.confidence))
        rgb = rgb_to_uint8_rgb(resp.rgb, node._bridge)
        depth = depth_to_meters_float(resp.depth, node._bridge)
        mask = mask_to_bool(best.mask, node._bridge)

        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise RuntimeError(
                f"RGB 与 depth 尺寸不一致: rgb={rgb.shape}, depth={depth.shape}"
            )
        if mask.shape[0] != rgb.shape[0] or mask.shape[1] != rgb.shape[1]:
            raise RuntimeError(
                f"mask 与 rgb 尺寸不一致: mask={mask.shape}, rgb={rgb.shape}"
            )

        np.savez_compressed(
            args.out,
            rgb=rgb.astype(np.uint8),
            depth=depth.astype(np.float32),
            ob_mask=mask.astype(bool),
            K=K.astype(np.float64),
            class_name=np.array(args.class_name),
            frame_id=np.array(ci.header.frame_id or ""),
        )
        node.get_logger().info(
            f"已写入 {args.out}  rgb={rgb.shape} depth={depth.shape} mask={mask.shape}"
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
