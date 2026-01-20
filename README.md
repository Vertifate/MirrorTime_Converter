# 4DGS 数据预处理管道

这是一个模块化的 4D 高斯散射(4DGS)数据预处理程序，用于将输入视频转换为可用于高斯训练的图片序列和相机参数。

#WDD [2026-01-19] [4DGS 预处理项目文档]

## 🎯 项目目标

从输入视频逐步处理为高斯训练所需的数据：
- 📹 提取高质量图像序列
- 📷 估计精确的相机参数（内参+外参）
- ✅ 验证数据质量
- 📦 输出标准格式（支持 Gaussian Splatting / NeRF）

## 🏗️ 架构设计

项目采用模块化管道架构，每个处理阶段是独立的模块：

```
视频输入 → 帧提取 → 图像预处理 → 相机估计 → 位姿精化 → 数据验证 → 输出格式化
```

### 核心模块

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| `video-input` | 视频验证和元数据提取 | OpenCV, ffmpeg |
| `frame-extraction` | 智能帧采样和去重 | 运动检测, 感知哈希 |
| `image-preprocessing` | 图像增强和校正 | 去模糊, 色彩校正 |
| `camera-estimation` | 相机参数估计 | COLMAP, SfM |
| `pose-refinement` | 位姿优化 | Bundle Adjustment |
| `data-validation` | 质量检查和报告 | 覆盖分析, 误差统计 |
| `output-formatter` | 格式转换和组织 | 多格式支持 |
| `pipeline-orchestrator` | 流程编排和调度 | 依赖管理, 缓存 |

## 🚀 快速开始

### 环境配置 (Ubuntu 22.04, Python 3.10, CUDA 11.8)

推荐使用 Conda 管理环境。以下步骤在 Ubuntu 22.04, CUDA 11.8 环境下测试通过：

```bash
# 1. 创建并激活 Conda 环境
conda create -n MirrorConverter python=3.10
conda activate MirrorConverter

# 2. 安装 PyTorch (CUDA 11.8)
pip3 install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu118


# 3. 安装项目依赖
pip install -r requirements.txt
```

### 其他依赖 (COLMAP)

```bash
# Ubuntu
sudo apt install colmap
# macOS
brew install colmap
```

### 运行完整流程

```bash
# 使用默认配置处理视频
python cli/main.py process --input video.mp4 --output ./output

# 使用高质量预设
python cli/main.py process --input video.mp4 --output ./output --preset high-quality

# 快速预览模式（低分辨率，少量帧）
python cli/main.py process --input video.mp4 --output ./output --preset fast-preview
```

### 交互式模式

```bash
python cli/interactive.py
```

## 📋 使用示例

### 命令行选项

```bash
python cli/main.py process \
  --input video.mp4 \
  --output ./output \
  --config custom_config.yaml \
  --stages video-input,frame-extraction,camera-estimation \
  --skip-cache \
  --verbose
```

### Python API

```python
from modules.pipeline_orchestrator.src.core.Pipeline import Pipeline
from core.config import load_config

# 加载配置
config = load_config("configs/presets/default.yaml")

# 创建流程
pipeline = Pipeline(config)

# 运行处理
result = pipeline.run(
    input_video="video.mp4",
    output_dir="./output"
)

# 检查结果
if result.success:
    print(f"处理完成！提取了 {result.num_frames} 帧")
    print(f"相机参数: {result.camera_params}")
else:
    print(f"处理失败: {result.error_message}")
```

## 📁 输出结构

```
output/
├── images/                 # 提取的图像序列
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── sparse/                 # COLMAP 稀疏重建结果
│   └── 0/
│       ├── cameras.txt     # 相机内参
│       ├── images.txt      # 图像和外参
│       └── points3D.txt    # 3D 点云
├── cameras/                # 相机参数（多种格式）
│   ├── transforms.json     # NeRF 格式
│   └── camera_params.yaml  # 自定义格式
├── metadata/               # 处理元数据
│   ├── video_info.json
│   ├── processing_log.txt
│   └── quality_metrics.json
└── reports/                # 验证报告
    └── quality_report.html
```

## ⚙️ 配置说明

### 预设配置

- `default.yaml` - 平衡质量和速度
- `high-quality.yaml` - 最高质量（慢）
- `fast-preview.yaml` - 快速预览（低质量）

### 自定义配置

复制模板并修改：

```bash
cp configs/templates/pipeline_template.yaml configs/my_config.yaml
# 编辑 my_config.yaml
python cli/main.py process --config configs/my_config.yaml --input video.mp4
```

关键配置项：

```yaml
stages:
  frame-extraction:
    config:
      target_frame_count: 300      # 目标帧数
      sampling_strategy: "motion_based"  # 采样策略
      
  camera-estimation:
    config:
      backend: "colmap"            # 后端选择
      colmap:
        feature_extractor:
          max_num_features: 8192   # 特征点数量
```

## 🔧 开发指南

### 添加新模块

```bash
# 使用生成器创建新模块
python scripts/generate-module.py my-new-stage --lang python

# 实现核心逻辑
# 1. 编辑 modules/my-new-stage/src/core/
# 2. 实现 IProcessor 接口
# 3. 添加测试
# 4. 更新配置模板
```

### 模块开发规范

每个处理模块必须：
1. 实现 `IProcessor` 接口
2. 提供完整的单元测试
3. 包含详细的 README.md
4. 定义清晰的输入输出契约

详见：[.agent/workflows/modular-architecture.md](.agent/workflows/modular-architecture.md)

## 📊 性能优化

### 缓存机制

系统自动缓存中间结果，避免重复计算：

```bash
# 清除缓存
python cli/main.py clear-cache

# 禁用缓存
python cli/main.py process --input video.mp4 --no-cache
```

### 并行处理

```yaml
global:
  num_workers: 8  # 增加并行 worker 数量
```

### GPU 加速

某些模块支持 GPU 加速（需要 CUDA）：

```yaml
camera-estimation:
  config:
    colmap:
      use_gpu: true
      gpu_index: 0
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest modules/frame-extraction/tests/

# 运行集成测试
pytest tests/integration/

# 端到端测试
pytest tests/e2e/
```

## 📚 文档

- [架构设计](docs/architecture.md) - 系统架构详解
- [流程说明](docs/pipeline-stages.md) - 各阶段详细说明
- [配置指南](docs/configuration-guide.md) - 完整配置参考
- [API 文档](docs/api-reference.md) - Python API 参考

## 🛠️ 依赖项

### 核心依赖

- Python 3.9+
- OpenCV
- NumPy
- PIL/Pillow
- PyYAML

### 可选依赖

- COLMAP（相机参数估计）
- FFmpeg（视频处理）
- CUDA（GPU 加速）

## 🤝 贡献

欢迎贡献！请遵循：

1. Fork 项目
2. 创建特性分支
3. 遵循代码规范
4. 添加测试
5. 提交 Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- COLMAP - 相机参数估计
- Gaussian Splatting - 目标训练框架
- NeRF - 数据格式参考

#WDD [2026-01-19] [项目 README 文档]