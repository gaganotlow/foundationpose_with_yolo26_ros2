# 在 Jetson 上运行 FoundationPose-TensorRT

适用于 NVIDIA Jetson 平台的快速设置指南。

## 系统要求

- Jetson 平台 (ARM64)
- CUDA 12.x (系统预装)
- TensorRT 10.x (系统预装)
- Python 3.10
- conda 环境: `foundationpose_ga`

## 快速开始

### 1. 安装依赖

```bash
conda activate foundationpose_ga
bash scripts/setup_jetson.sh
```

这个脚本会：
- ✅ 保留你现有的 PyTorch 和 TensorRT
- ✅ 只安装缺失的依赖包
- ✅ 不会修改系统环境

### 2. 编译 TensorRT 模型

```bash
source scripts/deps_jetson.sh && activate_deps
bash scripts/convert_onnx_jetson.sh
```

如果内存不足，可以降低 `chunk_size`：
```bash
# 编辑 scripts/convert_onnx_jetson.sh
chunk_size=128  # 默认是 252
```

### 3. 运行演示

```bash
source scripts/deps_jetson.sh && activate_deps
python demo.py
```

## 性能优化

### 降低输入分辨率

```python
cfg = FoundationPoseWrapperConfig(
    downsample_width=256,  # 降低到 256
    chunk_size=128,        # 如果重新编译了模型
)
```

### 启用最大性能模式

```bash
sudo nvpmodel -m 0      # 最大性能
sudo jetson_clocks      # 锁定最高频率
```

## 环境说明

所有 Jetson 脚本都是安全的：
- 只使用系统已安装的 CUDA/TensorRT
- 只修改当前 shell 的环境变量
- 不会修改系统配置
- 关闭终端后自动恢复

恢复环境：
```bash
deactivate_deps  # 或直接关闭终端
```

## 故障排除

### 内存不足
降低 `chunk_size` 到 64 或 32，重新编译模型。

### CUDA 找不到
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### 依赖冲突
脚本会自动检测已安装的包，不会重复安装或覆盖现有版本。
