# 单个检测实例：一个目标的全部数据
int32 instance_id
string class_name
float32 confidence
float32[4] bbox         # [x1, y1, x2, y2] 像素坐标
sensor_msgs/Image mask  # 与原图同尺寸的二值 mask（mono8，255=前景）
float32[3] center_point # mask 中心点对应的三维坐标 [x, y, z]，单位：米；若深度无效则全为 0

