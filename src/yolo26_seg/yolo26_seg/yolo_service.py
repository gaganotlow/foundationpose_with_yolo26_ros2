# -*- coding:utf-8 -*-

"""
YOLO26分割模型 TensorRT 推理 - ROS2 Service 版本
客户端传入类别名（如 "k2c"），服务端推理当前帧并返回该类别的检测结果（含3D中心点坐标）
"""

import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber

from yolo_interfaces.srv import YoloDetect
from yolo_interfaces.msg import YoloDetection
from yolo26_seg.yolo_node import YoloSegDetector


class YoloServiceNode(Node):

    def __init__(self):
        super().__init__('yolo26_service_node')

        self.declare_parameter('engine_file', '/media/rykj/nvme/jetson/ga/code/niusuo_perception/models/seg26_s_640_table.engine')
        self.declare_parameter('conf_thresh', 0.5)
        self.declare_parameter('service_name', 'yolo_detect')
        self.declare_parameter('input_topic', '/right_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/right_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/right_camera/color/camera_info')

        engine_file        = self.get_parameter('engine_file').value
        conf_thresh        = self.get_parameter('conf_thresh').value
        service_name       = self.get_parameter('service_name').value
        input_topic        = self.get_parameter('input_topic').value
        depth_topic        = self.get_parameter('depth_topic').value
        camera_info_topic  = self.get_parameter('camera_info_topic').value

        self.get_logger().info(f'正在加载模型: {engine_file}')
        self.detector = YoloSegDetector(
            engine_file=engine_file,
            conf_thresh=conf_thresh,
        )

        self.bridge = CvBridge()

        # 缓存最新帧
        self.latest_rgb      = None   # np.ndarray BGR
        self.latest_rgb_msg  = None   # sensor_msgs/Image
        self.latest_depth_msg = None  # sensor_msgs/Image

        # 相机内参（从 CameraInfo 话题获取一次后固定）
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_scale = 0.001  # Orbbec 深度单位：毫米 → 米

        # 订阅相机内参（只需一次，latched 兼容用 QoS depth=1）
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            1
        )

        # 订阅 RGB + Depth，时间同步缓存最新帧
        self.image_sub = Subscriber(self, Image, input_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=5,
            slop=0.05
        )
        self.ts.registerCallback(self._camera_callback)

        # 注册服务
        self.srv = self.create_service(YoloDetect, service_name, self._detect_callback)

        self.get_logger().info("=" * 60)
        self.get_logger().info("YOLO26 Service 节点已启动")
        self.get_logger().info(f"服务名称: /{service_name}")
        self.get_logger().info(f"订阅RGB:        {input_topic}")
        self.get_logger().info(f"订阅Depth:      {depth_topic}")
        self.get_logger().info(f"订阅CameraInfo: {camera_info_topic}")
        self.get_logger().info(f"支持类别: {YoloSegDetector.CLASS_NAMES}")
        self.get_logger().info(f"置信度阈值: {conf_thresh}")
        self.get_logger().info("=" * 60)

    # ------------------------------------------------------------------
    # 相机内参回调（只记录一次即可）
    # ------------------------------------------------------------------
    def _camera_info_callback(self, msg: CameraInfo):
        if self.fx is not None:
            return  # 已获取，不重复处理
        K = msg.k  # 行主序 3×3 内参矩阵
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        self.get_logger().info(
            f"获取相机内参: fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}"
        )

    # ------------------------------------------------------------------
    # 相机帧缓存回调
    # ------------------------------------------------------------------
    def _camera_callback(self, rgb_msg: Image, depth_msg: Image):
        self.latest_rgb_msg   = rgb_msg
        self.latest_depth_msg = depth_msg
        self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

    # ------------------------------------------------------------------
    # 工具：基于 mask 中心区域的深度中值计算三维坐标
    # ------------------------------------------------------------------
    def _mask_to_3d(self, mask_uint8: np.ndarray, depth_img: np.ndarray,
                    center_ratio: float = 0.5):
        """
        使用 mask 中心区域像素的深度中值计算三维坐标，比单点采样稳定。

        mask_uint8   : 与原图同尺寸的二值 mask（uint8，255=前景）
        depth_img    : uint16（毫米）或 float32（米）
        center_ratio : 仅使用距质心在 bbox 半径 * center_ratio 以内的像素，
                       过滤边缘噪声（默认 0.5，即内侧 50%）
        返回 (x, y, z) 米；无法获得有效深度时返回 (0, 0, 0)
        """
        if self.fx is None:
            return (0.0, 0.0, 0.0)

        ys, xs = np.where(mask_uint8 > 0)
        if len(xs) == 0:
            return (0.0, 0.0, 0.0)

        u_c = float(np.mean(xs))
        v_c = float(np.mean(ys))

        # 以 mask bbox 半径为基准，只保留靠近质心的内侧像素
        bbox_r = max(float(xs.max() - xs.min()),
                     float(ys.max() - ys.min())) / 2.0 * center_ratio
        bbox_r = max(bbox_r, 3.0)  # 最小 3 像素半径

        dist = np.sqrt((xs - u_c) ** 2 + (ys - v_c) ** 2)
        inner_mask = dist <= bbox_r
        xs_s = xs[inner_mask] if inner_mask.any() else xs
        ys_s = ys[inner_mask] if inner_mask.any() else ys

        # 批量采集深度并转米
        depths = depth_img[ys_s, xs_s].astype(np.float32)
        if depth_img.dtype == np.uint16:
            depths = depths * self.depth_scale  # mm → m

        valid = depths[(depths > 0.01) & np.isfinite(depths)]
        if valid.size == 0:
            return (0.0, 0.0, 0.0)

        z = float(np.median(valid))
        x = (u_c - self.cx) * z / self.fx
        y = (v_c - self.cy) * z / self.fy
        return (float(x), float(y), float(z))

    # ------------------------------------------------------------------
    # 服务回调
    # ------------------------------------------------------------------
    def _detect_callback(self, request, response):
        class_name = request.class_name.strip().lower()

        class_names_lower = [n.lower() for n in YoloSegDetector.CLASS_NAMES]
        if class_name not in class_names_lower:
            response.success = False
            response.message = f"未知类别 '{class_name}'，支持: {YoloSegDetector.CLASS_NAMES}"
            self.get_logger().warning(response.message)
            return response

        target_class_id = class_names_lower.index(class_name)

        if self.latest_rgb is None:
            response.success = False
            response.message = "尚未收到相机图像，请确认相机话题正常"
            self.get_logger().warning(response.message)
            return response

        self.get_logger().info(f"收到请求: class_name='{class_name}'，开始推理...")

        try:
            bboxes, masks = self.detector.inference(self.latest_rgb)

            response.header = self.latest_rgb_msg.header
            response.rgb    = self.latest_rgb_msg
            if self.latest_depth_msg is not None:
                response.depth = self.latest_depth_msg

            if bboxes is None or len(bboxes) == 0:
                response.success = False
                response.message = "当前帧未检测到任何目标"
                self.get_logger().info(response.message)
                return response

            # 预先将深度图转为 numpy，供 _pixel_to_3d 使用
            if self.latest_depth_msg is not None:
                depth_np = self.bridge.imgmsg_to_cv2(
                    self.latest_depth_msg, desired_encoding='passthrough'
                )
            else:
                depth_np = None

            instance_id = 0
            for bbox, mask in zip(bboxes, masks):
                x1, y1, x2, y2, conf, cid = bbox
                if int(cid) != target_class_id:
                    continue

                # mask 二值图
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
                mask_msg.header = self.latest_rgb_msg.header

                # 基于 mask 中心区域深度中值计算 3D 坐标
                if depth_np is not None:
                    cx3d, cy3d, cz3d = self._mask_to_3d(mask_uint8, depth_np)
                else:
                    cx3d = cy3d = cz3d = 0.0

                det = YoloDetection()
                det.instance_id  = instance_id
                det.class_name   = class_name
                det.confidence   = float(conf)
                det.bbox         = [float(x1), float(y1), float(x2), float(y2)]
                det.mask         = mask_msg
                det.center_point = [cx3d, cy3d, cz3d]

                response.detections.append(det)
                instance_id += 1

                self.get_logger().info(
                    f"  [{instance_id-1}] conf={conf:.2f} "
                    f"3D=({cx3d:.3f}, {cy3d:.3f}, {cz3d:.3f}) m"
                )

            count = len(response.detections)
            if count == 0:
                response.success = False
                response.message = f"当前帧未检测到类别 '{class_name}'"
            else:
                response.success = True
                response.message = f"检测到 {count} 个 '{class_name}' 目标"

            self.get_logger().info(response.message)

        except Exception as e:
            response.success = False
            response.message = f"推理出错: {str(e)}"
            self.get_logger().error(response.message)
            import traceback
            self.get_logger().error(traceback.format_exc())

        return response

    def destroy_node(self):
        self.get_logger().info("正在关闭YOLO26 Service节点...")
        if hasattr(self, 'detector'):
            self.detector.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
