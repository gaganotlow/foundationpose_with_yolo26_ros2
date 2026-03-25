#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoundationPose 服务调用客户端

用法：
  python3 fp_client.py                    # 默认检测 k2c，迭代次数用节点默认值
  python3 fp_client.py k2c               # 指定类别
  python3 fp_client.py k2c 5             # 指定类别 + 迭代次数
"""

import argparse
import sys
import rclpy
from rclpy.node import Node
from fp_interfaces.srv import FpDetect
from scipy.spatial.transform import Rotation
import numpy as np


class FpClient(Node):
    def __init__(self):
        super().__init__("fp_client")
        self._cli = self.create_client(FpDetect, "fp_detect")

    def call(self, class_name: str, est_refine_iter: int = 0, timeout: float = 120.0):
        self.get_logger().info(f"等待 /fp_detect 服务...")
        if not self._cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("服务不可用，请先启动 foundationpose 节点")
            return None

        req = FpDetect.Request()
        req.class_name = class_name
        req.est_refine_iter = est_refine_iter

        self.get_logger().info(f"发送请求: class_name='{class_name}'  （首次调用约需 30~60s）")
        future = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        if not future.done():
            self.get_logger().error(f"超时（{timeout}s），未收到响应")
            return None

        return future.result()


def print_result(resp):
    print("\n" + "=" * 60)
    if not resp.success:
        print(f"  失败: {resp.message}")
        print("=" * 60)
        return

    pose = resp.pose
    p = pose.pose.position
    q = pose.pose.orientation
    frame = pose.header.frame_id

    R = Rotation.from_quat([q.x, q.y, q.z, q.w])
    euler = R.as_euler("xyz", degrees=True)
    mat = R.as_matrix()

    print(f"  {resp.message}")
    print(f"  frame_id : {frame}")
    print("-" * 60)
    print(f"  平移 (m)  : x={p.x:+.4f}  y={p.y:+.4f}  z={p.z:+.4f}")
    print(f"  四元数    : x={q.x:+.4f}  y={q.y:+.4f}  z={q.z:+.4f}  w={q.w:+.4f}")
    print(f"  欧拉角(°) : roll={euler[0]:+.2f}  pitch={euler[1]:+.2f}  yaw={euler[2]:+.2f}")
    print("  旋转矩阵  :")
    for row in mat:
        print("    " + "  ".join(f"{v:+.4f}" for v in row))
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="调用 /fp_detect 服务获取物体 6D 位姿",
        usage="%(prog)s [class_name] [iter] [--timeout T]",
        epilog="例：python3 fp_client.py k2c 5",
    )
    parser.add_argument("class_name", nargs="?", default="k2c", help="目标类别（默认 k2c）")
    parser.add_argument("iter", nargs="?", type=int, default=0, help="精化迭代次数，0=节点默认值（默认 0）")
    parser.add_argument("--timeout", type=float, default=120.0, help="等待超时秒数（默认 120s）")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = FpClient()
    try:
        resp = node.call(args.class_name, args.iter, args.timeout)  # type: ignore[arg-type]
        if resp is not None:
            print_result(resp)
            return 0 if resp.success else 1
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
