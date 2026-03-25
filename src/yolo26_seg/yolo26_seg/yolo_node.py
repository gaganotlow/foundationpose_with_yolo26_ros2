import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from loguru import logger
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

class YoloSegDetector:
    
    CLASS_NAMES = ["j2", "k2c", "other", "j1", "v2"]
    
    CLASS_COLORS = {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (0, 255, 255)
    }
    
    def __init__(self, 
                engine_file="/media/rykj/nvme/jetson/ga/code/niusuo_perception/models/seg26_s_640_table.engine", 
                gpu_id=0,
                conf_thresh=0.5,
                kInputH=640,
                kInputW=640,
                ctx=None
            ):
        self.trt_file = engine_file
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.nums_classes = len(self.CLASS_NAMES)
        self.conf_thresh = conf_thresh
        self.kInputH = kInputH
        self.kInputW = kInputW
        self.mask_proto_dim = 32
        self.gpu_id = gpu_id
        self.ctx = ctx  # 保留参数以保持接口兼容性，但TensorRT 10不需要
        
        self.stream = cuda.Stream()
        self.engine = self.get_engine()

        self.context = self.engine.create_execution_context()
        
        self.tensor_names = []
        self.input_names = []
        self.output_names = []
        
        logger.info(f"Engine has {self.engine.num_io_tensors} IO tensors")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_names.append(name)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.context.set_input_shape(name, [1, 3, kInputH, kInputW])
                logger.info(f"Input tensor: {name}, shape: [1, 3, {kInputH}, {kInputW}]")
            else:
                self.output_names.append(name)
                logger.info(f"Output tensor: {name}")
        
        self.buffer_h = []
        self.buffer_d = []
        self.tensor_name_to_idx = {}
        
        for idx, name in enumerate(self.tensor_names):
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            host_mem = np.empty(shape, dtype=dtype)
            self.buffer_h.append(host_mem)
            
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.buffer_d.append(device_mem)
            
            self.context.set_tensor_address(name, int(device_mem))
            
            self.tensor_name_to_idx[name] = idx
            
            logger.info(f"Tensor[{idx}] '{name}': shape={shape}, dtype={dtype}, size={host_mem.size}")

        self._init_cuda_kernels()
        
    def release(self):
        try:
            if hasattr(self, 'context'):
                del self.context
            
            if hasattr(self, 'engine'):
                del self.engine
            
            if hasattr(self, 'buffer_d'):
                for buf in self.buffer_d:
                    try:
                        buf.free()
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error during release: {e}")
                   
    def get_engine(self):
        if os.path.exists(self.trt_file):
            with open(self.trt_file, "rb") as f:
                engine_string = f.read()
            if engine_string is None:
                print("Failed getting serialized engine!")
                return
            print("Succeeded getting serialized engine!")
        else:
            logger.info("Failed finding engine file!")
            exit(0)

        engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_string)

        return engine


    def _init_cuda_kernels(self):
        from pycuda.compiler import SourceModule
        
        mod = SourceModule("""
        #include <math.h>

        __global__ void letterbox_kernel_seg(
            const unsigned char* src, 
            float* dst, 
            int src_width, 
            int src_height, 
            int dst_width, 
            int dst_height,
            float scale,
            int pad_x,
            int pad_y,
            float norm_0,
            float norm_1,
            float norm_2,
            int is_bgr
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= dst_width || y >= dst_height) return;
            
            int src_x = (int)((x - pad_x) / scale);
            int src_y = (int)((y - pad_y) / scale);
            
            src_x = max(0, min(src_x, src_width - 1));
            src_y = max(0, min(src_y, src_height - 1));
            
            int dst_idx = y * dst_width + x;
            
            int src_idx;

            src_idx = (src_y * src_width + src_x) * 3;
            
            if (x >= pad_x && x < dst_width - pad_x && y >= pad_y && y < dst_height - pad_y) {
                if (is_bgr) {
                    dst[dst_idx] = (src[src_idx + 2] / 255.0f - norm_0) / norm_1;
                    dst[dst_idx + dst_width * dst_height] = (src[src_idx + 1] / 255.0f - norm_0) / norm_1;
                    dst[dst_idx + 2 * dst_width * dst_height] = (src[src_idx] / 255.0f - norm_0) / norm_1;
                } else {
                    dst[dst_idx] = (src[src_idx] / 255.0f - norm_0) / norm_1;
                    dst[dst_idx + dst_width * dst_height] = (src[src_idx + 1] / 255.0f - norm_0) / norm_1;
                    dst[dst_idx + 2 * dst_width * dst_height] = (src[src_idx + 2] / 255.0f - norm_0) / norm_1;
                }
            } else {
                dst[dst_idx] = (114.0f / 255.0f - norm_0) / norm_1;
                dst[dst_idx + dst_width * dst_height] = (114.0f / 255.0f - norm_0) / norm_1;
                dst[dst_idx + 2 * dst_width * dst_height] = (114.0f / 255.0f - norm_0) / norm_1;
            }
        }
        
        __global__ void mask_decode_kernel(
            const float* mask_proto,
            const float* mask_coeff,
            float* output_masks,
            int num_masks,
            int proto_h,
            int proto_w)
        {
            const int n = blockIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const int x = blockIdx.z * blockDim.z + threadIdx.z;
            
            if (n >= num_masks || y >= proto_h || x >= proto_w) return;
            
            float sum = 0.0f;
            for (int k = 0; k < 32; ++k) {
                sum += mask_coeff[n*32 + k] * mask_proto[k*proto_h*proto_w + y*proto_w + x];
            }
            output_masks[n*proto_h*proto_w + y*proto_w + x] = 1.0f / (1.0f + expf(-sum));
        }      

                
        """)
        
        self.letterbox_kernel = mod.get_function("letterbox_kernel_seg")
        self.mask_decode = mod.get_function("mask_decode_kernel")
        logger.info("CUDA kernels initialized.")
 
    def inference_one(self, data_input, context, buffer_h, buffer_d):
        input_idx = self.tensor_name_to_idx[self.input_names[0]]
        
        buffer_h[input_idx] = np.ascontiguousarray(data_input)
        
        cuda.memcpy_htod_async(buffer_d[input_idx], buffer_h[input_idx], self.stream)
        
        context.execute_async_v3(stream_handle=self.stream.handle)
        
        output_indices = [self.tensor_name_to_idx[name] for name in self.output_names]
        for idx in output_indices:
            cuda.memcpy_dtoh_async(buffer_h[idx], buffer_d[idx], self.stream)
        self.stream.synchronize()

        outputs = []
        for i, name in enumerate(self.output_names):
            idx = output_indices[i]
            output_shape = buffer_h[idx].shape
            logger.debug(f"Output {name}: shape={output_shape}, size={buffer_h[idx].size}")
            
            if len(output_shape) == 4 and output_shape[1] == 32:
                # Mask proto: [1, 32, 200, 200]
                proto = buffer_h[idx].copy().squeeze(0)
                outputs.append(('proto', proto))
            elif len(output_shape) == 3 and output_shape[1] == 300:
                # Detection output (end2end): [1, 300, 38]
                det = buffer_h[idx].copy().squeeze(0)
                outputs.append(('det', det))
            else:
                logger.warning(f"Unknown output shape: {output_shape}")
                outputs.append(('unknown', buffer_h[idx].copy()))
        
        proto = None
        det = None
        for output_type, data in outputs:
            if output_type == 'proto':
                proto = data
            elif output_type == 'det':
                det = data
        
        if proto is None or det is None:
            logger.error(f"Failed to identify outputs. Got: {[(t, d.shape) for t, d in outputs]}")
            if len(outputs) >= 2:
                proto = outputs[0][1] if outputs[0][1].ndim == 3 else outputs[1][1]
                det = outputs[1][1] if outputs[1][1].ndim == 2 else outputs[0][1]
        
        return proto, det
  

    def preprocess_with_cuda(self, image: np.ndarray) -> np.ndarray:
        src_height, src_width = image.shape[:2]
        dst_height, dst_width = self.kInputH, self.kInputW
        
        scale = min(dst_width / src_width, dst_height / src_height)
        new_width = int(src_width * scale)
        new_height = int(src_height * scale)
        pad_x = (dst_width - new_width) // 2
        pad_y = (dst_height - new_height) // 2
        
        src_device = cuda.mem_alloc(image.nbytes)
        dst_device = cuda.mem_alloc(3 * dst_width * dst_height * np.float32().itemsize)
        
        cuda.memcpy_htod(src_device, image)
        
        block = (16, 16, 1)
        grid = ((dst_width + block[0] - 1) // block[0], (dst_height + block[1] - 1) // block[1], 1)
        
        self.letterbox_kernel(
            src_device, dst_device,
            np.int32(src_width), np.int32(src_height),
            np.int32(dst_width), np.int32(dst_height),
            np.float32(scale),
            np.int32(pad_x), np.int32(pad_y),
            np.float32(0.0), np.float32(1.0), np.float32(1.0),
            np.int32(1),
            block=block, grid=grid
        )
        
        output = np.empty((3, dst_height, dst_width), dtype=np.float32)
        cuda.memcpy_dtoh(output, dst_device)
        
        src_device.free()
        dst_device.free()
        
        return np.expand_dims(output, axis=0)

    @staticmethod
    def _nms(boxes_xyxy, scores, cls_ids, iou_thresh=0.3):
        """
        跨类别 IoU NMS：按置信度全局排序，重叠框无论是否同类都压制。
        解决同一物体被同时识别为多个类别的问题。
        返回保留的索引数组。
        """
        if len(boxes_xyxy) == 0:
            return np.array([], dtype=np.int32)

        x1, y1, x2, y2 = (boxes_xyxy[:, i] for i in range(4))
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

        # 全局按置信度降序排列
        order = np.argsort(scores)[::-1]
        suppressed = np.zeros(len(order), dtype=bool)
        keep_indices = []

        for i in range(len(order)):
            if suppressed[i]:
                continue
            cur = order[i]
            keep_indices.append(cur)

            rest = order[i + 1:]
            ix1 = np.maximum(x1[cur], x1[rest])
            iy1 = np.maximum(y1[cur], y1[rest])
            ix2 = np.minimum(x2[cur], x2[rest])
            iy2 = np.minimum(y2[cur], y2[rest])
            inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
            iou = inter / (areas[cur] + areas[rest] - inter + 1e-6)
            suppressed[i + 1:] |= (iou > iou_thresh)

        return np.array(keep_indices, dtype=np.int32)

    def _gpu_mask_decode(self, proto, coeff):
        num_masks = coeff.shape[0]
        proto_h, proto_w = proto.shape[1], proto.shape[2]
        output = np.zeros((num_masks, proto_h, proto_w), dtype=np.float32)
        
        if num_masks == 0:
            return None
        
        proto_gpu = cuda.mem_alloc(proto.nbytes)
        cuda.memcpy_htod(proto_gpu, proto)
        coeff_gpu = cuda.mem_alloc(coeff.nbytes)
        cuda.memcpy_htod(coeff_gpu, coeff)
        output_gpu = cuda.mem_alloc(output.nbytes)
        
        block = (1, 16, 16)
        grid = (
            num_masks,
            (proto_h + block[1]-1) // block[1],
            (proto_w + block[2]-1) // block[2]
        )
        self.mask_decode(
            proto_gpu, coeff_gpu, output_gpu,
            np.int32(num_masks), np.int32(proto_h), np.int32(proto_w),
            block=block, grid=grid)
        
        cuda.memcpy_dtoh(output, output_gpu)
        
        proto_gpu.free()
        coeff_gpu.free()
        output_gpu.free()
        
        return output
    
    def postprocess_with_cuda(self, outputs, orig_shape):
        """
        使用PyCUDA加速的后处理 (YOLO26 end2end模式)
        YOLO26 end2end输出已经做过NMS，不需要再次NMS
        输出格式: [N, 38] = [x1, y1, x2, y2, score, class_id, mask_coef_1...mask_coef_32]
        注意：end2end模式输出的bbox坐标已经是xyxy格式（相对于输入尺寸如640x640）
        """
        proto, pred = outputs[0], outputs[1]

        logger.debug(f"pred.shape: {pred.shape}")    # [300, 38]
        logger.debug(f"proto.shape: {proto.shape}")  # [32, 200, 200]

        # YOLO26 end2end输出格式: [N, 38]
        # 前4维: bbox坐标 (x1, y1, x2, y2) - 已经是xyxy格式
        # 第5维: 置信度 score
        # 第6维: 类别ID class_id
        # 后32维: mask系数 mask_coefficients
        boxes_xyxy = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        cls_ids = pred[:, 5].astype(np.float32)
        mask_coeff = pred[:, 6:].astype(np.float32)

        valid_mask = scores > self.conf_thresh

        if not np.any(valid_mask):
            logger.debug("No detections after confidence filtering")
            return None, None

        boxes_xyxy = boxes_xyxy[valid_mask]
        scores = scores[valid_mask]
        cls_ids = cls_ids[valid_mask]
        mask_coeff = mask_coeff[valid_mask]

        # 补充 IoU NMS，过滤模型内置 NMS 未能滤掉的重叠框
        keep = self._nms(boxes_xyxy, scores, cls_ids, iou_thresh=0.3)
        if len(keep) == 0:
            return None, None
        boxes_xyxy = boxes_xyxy[keep]
        scores     = scores[keep]
        cls_ids    = cls_ids[keep]
        mask_coeff = mask_coeff[keep]

        logger.debug(f"After NMS: {len(boxes_xyxy)} detections")

        if len(boxes_xyxy) > 0:
            logger.debug(f"First box (before scale): {boxes_xyxy[0]}, score: {scores[0]:.3f}, class: {int(cls_ids[0])}")

        bboxes = np.column_stack((boxes_xyxy, scores, cls_ids)).astype(np.float32)

        # end2end模式已经做过NMS，直接解码mask
        masks = self._gpu_mask_decode(proto, mask_coeff)
        if masks is None or masks.shape[0] == 0:
            return None, None

        keep_boxes = bboxes.copy()
        keep_mask_boxes = bboxes.copy()

        keep_boxes[:, :4] = self._scale_coords((self.kInputH, self.kInputW), keep_boxes[:, :4], orig_shape)

        if len(keep_boxes) > 0:
            logger.debug(f"First box (after scale): {keep_boxes[0][:4]}, orig_shape: {orig_shape}")

        masks = self.process_mask(masks, keep_mask_boxes, proto.shape, (self.kInputH, self.kInputW))
        final_masks = self.scale_masks(masks, orig_shape)

        return keep_boxes, final_masks

    def process_mask(self, masks, bboxes, proto_shape, input_shape, upsample=False):
        
        c, mh, mw = proto_shape
        ih, iw = input_shape

        width_ratio = mw / iw
        height_ratio = mh / ih

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= width_ratio
        downsampled_bboxes[:, 1] *= height_ratio
        downsampled_bboxes[:, 2] *= width_ratio
        downsampled_bboxes[:, 3] *= height_ratio

        bbox_w = downsampled_bboxes[:, 2] - downsampled_bboxes[:, 0]
        bbox_h = downsampled_bboxes[:, 3] - downsampled_bboxes[:, 1]
        expand_w = bbox_w * 0.1 / 2
        expand_h = bbox_h * 0.1 / 2
        downsampled_bboxes[:, 0] -= expand_w
        downsampled_bboxes[:, 1] -= expand_h
        downsampled_bboxes[:, 2] += expand_w
        downsampled_bboxes[:, 3] += expand_h
        
        downsampled_bboxes[:, 0] = np.maximum(downsampled_bboxes[:, 0], 0)
        downsampled_bboxes[:, 1] = np.maximum(downsampled_bboxes[:, 1], 0)
        downsampled_bboxes[:, 2] = np.minimum(downsampled_bboxes[:, 2], mw)
        downsampled_bboxes[:, 3] = np.minimum(downsampled_bboxes[:, 3], mh)

        masks = self.crop_mask(masks, downsampled_bboxes)
        if upsample:
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (iw, ih), interpolation=cv2.INTER_LINEAR)
            if len(masks.shape) == 2:
                masks = masks[:, :, None]
            masks = masks.transpose((2, 0, 1))

        return masks        

    def crop_mask(self, masks, boxes):
        _, h, w = masks.shape
        x1 = boxes[:, 0][:, None, None]
        y1 = boxes[:, 1][:, None, None]
        x2 = boxes[:, 2][:, None, None]
        y2 = boxes[:, 3][:, None, None]

        c = np.arange(w, dtype=x1.dtype)[None, None, :]
        r = np.arange(h, dtype=x1.dtype)[None, :, None]

        return masks * ((c >= x1) * (c < x2) * (r >= y1) * (r < y2))    
    
    def scale_masks(self, masks, img0_shape):
        masks = masks.transpose((1, 2, 0))
        img1_shape = masks.shape
        if img1_shape[:2] == img0_shape[:2]:
            masks = masks.transpose((2, 0, 1))
            return masks
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

        top, left = int(pad[1]), int(pad[0])
        bottom, right = int(img1_shape[0] - pad[1]), int(img1_shape[1] - pad[0])

        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (img0_shape[1], img0_shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]

        masks = masks.transpose((2, 0, 1))

        mask1 = masks > 0.5
        mask2 = masks <= 0.5
        masks[mask1] = 1
        masks[mask2] = 0

        return masks
    
    def _scale_coords(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[0] - img0_shape[0] * gain) / 2, (img1_shape[1] - img0_shape[1] * gain) / 2

        coords[:, [0, 2]] -= pad[1]
        coords[:, [1, 3]] -= pad[0]
        coords[:, :4] /= gain

        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])

        return coords
    
        
    def inference(self, image):
        """推理方法 - TensorRT 10"""
        if image is None:
            return None, None
        orig_shape = image.shape[:2]

        t0 = time.time()
        input_data = self.preprocess_with_cuda(image)
        t1 = time.time()
        preprocess_time = (t1 - t0) * 1000

        output = self.inference_one(input_data, self.context, self.buffer_h, self.buffer_d)
        t2 = time.time()
        inference_time = (t2 - t1) * 1000

        bboxes, masks = self.postprocess_with_cuda(output, orig_shape)
        t3 = time.time()
        postprocess_time = (t3 - t2) * 1000

        total_time = (t3 - t0) * 1000

        logger.info(f"耗时统计: 预处理={preprocess_time:.2f}ms | 推理={inference_time:.2f}ms | 后处理={postprocess_time:.2f}ms | 总计={total_time:.2f}ms ({1000/total_time:.1f} FPS)")

        if bboxes is None:
            return None, None
        return bboxes, masks

    def get_detection_data(self, image):
        bboxes, masks = self.inference(image)
        return bboxes, masks


def draw_segmentation_result(image, bboxes, masks, class_names_list=None, colors=None):
    if bboxes is None or len(bboxes) == 0:
        return image
    
    if class_names_list is None:
        class_names_list = YoloSegDetector.CLASS_NAMES
    if colors is None:
        colors = YoloSegDetector.CLASS_COLORS
    
    result_img = image.copy()
    
    for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
        x1, y1, x2, y2, conf, class_id = bbox
        class_id = int(class_id)
        
        color = colors.get(class_id, (255, 255, 255))
        
        mask_bool = mask.astype(bool)
        mask_color = np.zeros(result_img.shape, dtype=np.uint8)
        mask_color[mask_bool] = color
        
        result_img[mask_bool] = result_img[mask_bool] * 0.5 + mask_color[mask_bool] * 0.5
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf, class_id = bbox
        class_id = int(class_id)
        color = colors.get(class_id, (255, 255, 255))
        
        cv2.rectangle(result_img, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     color, 
                     thickness=2, 
                     lineType=cv2.LINE_AA)
        
        label = f"{class_names_list[class_id]} {conf:.2f}"
        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
        c1 = (int(x1), int(y1))
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        
        cv2.rectangle(result_img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(result_img, label, 
                   (c1[0], c1[1] - 2), 
                   0, 0.6, (255, 255, 255),
                   thickness=2, lineType=cv2.LINE_AA)
    
    return result_img


def save_individual_masks(masks, bboxes, output_dir, img_name, class_names_list=None):
    if masks is None or len(masks) == 0:
        return
    
    if class_names_list is None:
        class_names_list = YoloSegDetector.CLASS_NAMES
    
    for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
        class_id = int(bbox[5])
        class_name = class_names_list[class_id]
        conf = bbox[4]
        
        mask_binary = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(output_dir, f"{img_name}_{class_name}_{i}_conf{conf:.2f}_mask.png")
        cv2.imwrite(mask_path, mask_binary)
        
        logger.info(f"保存mask: {mask_path}")


class YoloROS2Node(Node):
    
    def __init__(self):
        super().__init__('yolo26_seg_node')

        self.declare_parameter('engine_file', '/media/rykj/nvme/jetson/ga/code/cuda_model/seg26_s_640_table.engine')
        self.declare_parameter('input_topic', '/right_camera/color/image_raw')
        self.declare_parameter('output_topic', '/yolo26/result_image')
        self.declare_parameter('camera_info_topic', '/right_camera/color/camera_info')
        self.declare_parameter('conf_thresh', 0.5)
        self.declare_parameter('save_results', False)
        self.declare_parameter('output_dir', '/media/rykj/nvme/jetson/ga/code/yolo26_seg/output_ros')
        
        engine_file = self.get_parameter('engine_file').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        conf_thresh = self.get_parameter('conf_thresh').value
        self.save_results = self.get_parameter('save_results').value
        self.output_dir = self.get_parameter('output_dir').value
        
        self.get_logger().info(f'正在加载模型: {engine_file}')
        self.detector = YoloSegDetector(
            engine_file=engine_file,
            conf_thresh=conf_thresh
        )
        
        self.bridge = CvBridge()
        
        self.camera_info = None
        
        self.image_sub = Subscriber(self, Image, input_topic)
        self.camera_info_sub = Subscriber(self, CameraInfo, camera_info_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.camera_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        self.result_pub = self.create_publisher(
            Image,
            output_topic,
            10
        )
        
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        self.get_logger().info("="*60)
        self.get_logger().info("YOLO26分割节点已启动")
        self.get_logger().info(f"订阅话题: {input_topic}")
        self.get_logger().info(f"发布话题: {output_topic}")
        self.get_logger().info(f"相机内参话题: {camera_info_topic}")
        self.get_logger().info(f"类别: {YoloSegDetector.CLASS_NAMES}")
        self.get_logger().info(f"置信度阈值: {conf_thresh}")
        self.get_logger().info(f"保存结果: {self.save_results}")
        if self.save_results:
            self.get_logger().info(f"输出目录: {self.output_dir}")
        self.get_logger().info("="*60)
    
    
    def synchronized_callback(self, image_msg, camera_info_msg):
        try:
            if self.camera_info is None:
                self.camera_info = camera_info_msg
                self.get_logger().info("="*60)
                self.get_logger().info(f'接收到相机内参（已同步）')
                self.get_logger().info(f'分辨率: {camera_info_msg.width}x{camera_info_msg.height}')
                self.get_logger().info(f'畸变模型: {camera_info_msg.distortion_model}')
                self.get_logger().info(f'内参矩阵K:')
                self.get_logger().info(f'  fx: {camera_info_msg.k[0]:.2f}, fy: {camera_info_msg.k[4]:.2f}')
                self.get_logger().info(f'  cx: {camera_info_msg.k[2]:.2f}, cy: {camera_info_msg.k[5]:.2f}')
                self.get_logger().info(f'frame_id: {camera_info_msg.header.frame_id}')
                self.get_logger().info("="*60)
            
            self.camera_info = camera_info_msg
            
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            start_time = time.time()
            
            bboxes, masks = self.detector.inference(cv_image)
            
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1
            
            if bboxes is not None and len(bboxes) > 0:
                result_img = draw_segmentation_result(cv_image, bboxes, masks)
                
                fps = 1.0 / inference_time if inference_time > 0 else 0
                avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
                info_text = f"FPS: {fps:.1f} | Avg: {avg_fps:.1f} | Dets: {len(bboxes)}"
                cv2.putText(result_img, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if self.frame_count % 30 == 0:
                    self.get_logger().info(f"Frame {self.frame_count}: 检测到 {len(bboxes)} 个目标, FPS: {fps:.1f}")
                    for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2, conf, class_id = bbox
                        class_name = YoloSegDetector.CLASS_NAMES[int(class_id)]
                        self.get_logger().info(f"  [{i}] {class_name}: conf={conf:.3f}")
                
                if self.save_results and self.frame_count % 100 == 0:
                    timestamp = image_msg.header.stamp.sec * 1000000000 + image_msg.header.stamp.nanosec
                    img_name = f"frame_{self.frame_count}_{timestamp}"
                    result_path = os.path.join(self.output_dir, f"{img_name}_result.jpg")
                    cv2.imwrite(result_path, result_img)
                    save_individual_masks(masks, bboxes, self.output_dir, img_name)
            else:
                result_img = cv_image
                if self.frame_count % 30 == 0:
                    self.get_logger().info(f"Frame {self.frame_count}: 未检测到目标")
            
            result_msg = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
            result_msg.header = image_msg.header
            self.result_pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f'处理同步消息时出错: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def destroy_node(self):
        self.get_logger().info("正在关闭YOLO26节点...")
        if hasattr(self, 'detector'):
            self.detector.release()
        self.get_logger().info(f"处理了 {self.frame_count} 帧图像")
        if self.frame_count > 0:
            avg_fps = self.frame_count / self.total_inference_time
            self.get_logger().info(f"平均FPS: {avg_fps:.2f}")
        super().destroy_node()


def main():
    rclpy.init()

    node = YoloROS2Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_offline():
    """
    离线单图推理模式（不依赖ROS2）。
    如需离线测试，将 __main__ 中的 main() 替换为 main_offline() 即可。
    """
    image_path = "/media/rykj/nvme/jetson/ga/code/FoundationPose/demo_data/k2c/rgb/0.png"
    output_path = "/media/rykj/nvme/jetson/ga/code/yolo26_seg/output_offline/result.jpg"

    detector = YoloSegDetector(
        engine_file="/media/rykj/nvme/jetson/ga/code/cuda_model/seg26_s_640_table.engine",
        conf_thresh=0.5,
    )

    try:
        image = cv2.imread(image_path)
        bboxes, masks = detector.inference(image)

        if bboxes is not None and len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2, conf, class_id = bbox
                logger.info(f"[{i}] {YoloSegDetector.CLASS_NAMES[int(class_id)]}: conf={conf:.3f}, bbox=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_img = draw_segmentation_result(image, bboxes, masks)
            cv2.imwrite(output_path, result_img)
            logger.info(f"结果已保存: {output_path}")
        else:
            logger.warning("未检测到任何目标")
    finally:
        detector.release()


if __name__ == "__main__":
    # ROS2相机模式（由launch文件启动，参数通过launch传入）
    main()

    # 离线测试模式（注释掉上面的main()，取消注释下面这行即可）
    # main_offline()
