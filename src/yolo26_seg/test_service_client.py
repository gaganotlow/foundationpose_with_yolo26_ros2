#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_detect 服务测试客户端（完整输出版）
用法:
    python3 test_service_client.py [class_name]

图片保存目录：相对当前工作目录的 yolo_output/（无需传参）
请在期望生成 yolo_output 的目录下执行本脚本，例如:
    cd ~/your_ws && python3 src/yolo26_seg/test_service_client.py k2c
"""

import os
import sys
import datetime
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from yolo_interfaces.srv import YoloDetect

# 相对当前工作目录，不写绝对路径
OUTPUT_DIR = 'yolo_output'


class YoloDetectClient(Node):

    def __init__(self):
        super().__init__('yolo_detect_client')
        self.cli = self.create_client(YoloDetect, 'yolo_detect')
        self.bridge = CvBridge()

        timeout = 5.0
        if not self.cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"服务 /yolo_detect 未就绪（等待 {timeout}s 超时）")
            sys.exit(1)

        self.get_logger().info("已连接到 /yolo_detect 服务")

    def send_request(self, class_name: str):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        req = YoloDetect.Request()
        req.class_name = class_name

        self.get_logger().info(f"发送请求: class_name='{class_name}'")
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        resp = future.result()
        if resp is None:
            self.get_logger().error("服务调用失败，返回 None")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        print("=" * 70)
        print(f"success      : {resp.success}")
        print(f"message      : {resp.message}")
        print(f"header stamp : {resp.header.stamp.sec}.{resp.header.stamp.nanosec:09d}")
        print(f"header frame : {resp.header.frame_id}")

        # ── RGB 图 ──────────────────────────────────────────────
        if resp.rgb.width > 0:
            rgb = self.bridge.imgmsg_to_cv2(resp.rgb, desired_encoding='bgr8')
            print(f"rgb          : {resp.rgb.width}x{resp.rgb.height}  encoding={resp.rgb.encoding}")
            rgb_path = os.path.join(OUTPUT_DIR, f"{ts}_rgb.png")
            cv2.imwrite(rgb_path, rgb)
            print(f"  → 保存: {rgb_path}")
        else:
            rgb = None
            print("rgb          : (未返回)")

        # ── Depth 图 ────────────────────────────────────────────
        if resp.depth.width > 0:
            depth_raw = self.bridge.imgmsg_to_cv2(resp.depth, desired_encoding='passthrough')
            print(f"depth        : {resp.depth.width}x{resp.depth.height}  encoding={resp.depth.encoding}")
            # 保存原始深度（16bit PNG）
            depth_path = os.path.join(OUTPUT_DIR, f"{ts}_depth.png")
            cv2.imwrite(depth_path, depth_raw)
            print(f"  → 保存原始: {depth_path}")
            # 同时保存可视化深度（归一化伪彩色）
            depth_vis = cv2.normalize(depth_raw.astype(np.float32), None, 0, 255,
                                      cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis_path = os.path.join(OUTPUT_DIR, f"{ts}_depth_vis.png")
            cv2.imwrite(depth_vis_path, depth_vis)
            print(f"  → 保存伪彩色: {depth_vis_path}")
        else:
            depth_raw = None
            print("depth        : (未返回)")

        # ── 检测结果 ─────────────────────────────────────────────
        print(f"\n检测数量     : {len(resp.detections)}")

        # 在 RGB 上叠加所有结果
        vis = rgb.copy() if rgb is not None else None

        for det in resp.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            px, py, pz = det.center_point

            print("-" * 70)
            print(f"  instance_id  : {det.instance_id}")
            print(f"  class_name   : {det.class_name}")
            print(f"  confidence   : {det.confidence:.4f}")
            print(f"  bbox         : [{x1}, {y1}, {x2}, {y2}]  "
                  f"w={x2-x1}px  h={y2-y1}px")
            print(f"  center 3D    : x={px:.4f}m  y={py:.4f}m  z={pz:.4f}m")

            # mask 信息及保存
            mask = self.bridge.imgmsg_to_cv2(det.mask, desired_encoding='mono8')
            fg_pixels = int(np.count_nonzero(mask))
            print(f"  mask size    : {det.mask.width}x{det.mask.height}")
            print(f"  mask fg px   : {fg_pixels}  "
                  f"({100.0*fg_pixels/(det.mask.width*det.mask.height):.1f}%)")

            mask_path = os.path.join(OUTPUT_DIR,
                                     f"{ts}_mask_{det.instance_id}_{det.class_name}.png")
            cv2.imwrite(mask_path, mask)
            print(f"  → mask 保存: {mask_path}")

            # 在可视化图上叠加
            if vis is not None:
                color = (0, 255, 0) if det.class_name == 'k2c' else \
                        (0, 128, 255) if det.class_name == 'j2' else (255, 0, 0)
                overlay = vis.copy()
                overlay[mask > 0] = (
                    overlay[mask > 0] * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)
                vis = overlay
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} {det.confidence:.2f} z={pz:.2f}m"
                cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存叠加可视化结果
        if vis is not None:
            vis_path = os.path.join(OUTPUT_DIR, f"{ts}_result.png")
            cv2.imwrite(vis_path, vis)
            print("-" * 70)
            print(f"\n叠加结果保存: {vis_path}")

        print("=" * 70)
        print(f"所有文件保存至: {OUTPUT_DIR}/")


def main():
    rclpy.init()

    class_name = sys.argv[1] if len(sys.argv) > 1 else 'k2c'

    client = YoloDetectClient()
    client.send_request(class_name)
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
