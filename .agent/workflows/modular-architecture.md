---
description: 模块化架构设计指南 - 专为 4DGS 数据预处理管道定制
---

# 模块化架构设计 Skill - 4DGS 数据处理专用版

这个 skill 专为 **4DGS 高斯数据预处理程序** 定制，帮助您构建一个模块化的数据处理管道，从视频输入到高斯训练数据输出的完整流程。

> 💡 **项目目标**: 将输入视频逐步处理为用于高斯训练的图片序列和摄像机参数

## 🎯 4DGS 数据处理管道概览

```
视频输入 → 帧提取 → 图像预处理 → 相机参数估计 → 数据验证 → 输出组织
   │          │          │              │              │           │
   └──────────┴──────────┴──────────────┴──────────────┴───────────┘
                      独立的处理模块（Pipeline Stages）
```

## 📋 推荐的模块划分

基于 4DGS 预处理的典型工作流，建议以下模块划分：

1. **video-input** - 视频输入与验证
2. **frame-extraction** - 帧提取与采样
3. **image-preprocessing** - 图像预处理（去模糊、色彩校正等）
4. **camera-estimation** - 相机参数估计（COLMAP/SfM）
5. **pose-refinement** - 相机位姿精化
6. **data-validation** - 数据质量验证
7. **output-formatter** - 输出格式化（生成训练所需的目录结构）
8. **pipeline-orchestrator** - 流程编排器（串联所有阶段）

## 核心原则

### 1. 模块独立性
- **高内聚低耦合**: 每个模块内部紧密关联，模块之间松散耦合
- **单一职责**: 每个模块专注于一个特定功能领域
- **接口隔离**: 模块通过定义良好的接口进行交互
- **依赖最小化**: 减少模块间的依赖关系


### 2. 4DGS 数据处理管道目录结构

针对您的 4DGS 预处理项目，推荐以下目录结构：

```
4dgs-data-preprocessor/
├── modules/                           # 处理阶段模块
│   ├── video-input/                  # 视频输入模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── VideoReader.py    # 视频读取器
│   │   │   │   ├── VideoValidator.py # 视频验证器
│   │   │   │   └── MetadataExtractor.py # 元数据提取
│   │   │   ├── models/
│   │   │   │   └── VideoInfo.py      # 视频信息数据模型
│   │   │   └── index.py              # 模块入口
│   │   ├── tests/
│   │   ├── config/
│   │   │   └── video_config.yaml     # 支持的视频格式配置
│   │   └── README.md
│   │
│   ├── frame-extraction/              # 帧提取模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── FrameExtractor.py  # 帧提取器
│   │   │   │   ├── FrameSampler.py    # 采样策略（均匀/智能）
│   │   │   │   └── FrameDeduplicator.py # 去重
│   │   │   ├── strategies/            # 采样策略
│   │   │   │   ├── UniformSampler.py  # 均匀采样
│   │   │   │   ├── KeyframeSampler.py # 关键帧采样
│   │   │   │   └── MotionBasedSampler.py # 基于运动采样
│   │   │   └── index.py
│   │   ├── tests/
│   │   └── README.md
│   │
│   ├── image-preprocessing/           # 图像预处理模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── ImageEnhancer.py   # 图像增强
│   │   │   │   ├── DeblurFilter.py    # 去模糊
│   │   │   │   └── ColorCorrector.py  # 色彩校正
│   │   │   ├── filters/               # 各种滤镜
│   │   │   └── index.py
│   │   └── README.md
│   │
│   ├── camera-estimation/             # 相机参数估计模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── ColmapWrapper.py   # COLMAP 封装
│   │   │   │   ├── SfMProcessor.py    # SfM 处理器
│   │   │   │   └── IntrinsicEstimator.py # 内参估计
│   │   │   ├── backends/              # 不同后端支持
│   │   │   │   ├── colmap_backend.py
│   │   │   │   ├── opensfm_backend.py
│   │   │   │   └── metashape_backend.py
│   │   │   └── index.py
│   │   └── README.md
│   │
│   ├── pose-refinement/               # 位姿精化模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── PoseOptimizer.py   # 位姿优化器
│   │   │   │   ├── BundleAdjustment.py # BA 优化
│   │   │   │   └── OutlierRemoval.py  # 异常值去除
│   │   │   └── index.py
│   │   └── README.md
│   │
│   ├── data-validation/               # 数据验证模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── QualityChecker.py  # 质量检查
│   │   │   │   ├── CoverageAnalyzer.py # 覆盖分析
│   │   │   │   └── ReportGenerator.py # 报告生成
│   │   │   └── index.py
│   │   └── README.md
│   │
│   ├── output-formatter/              # 输出格式化模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── DataOrganizer.py   # 数据组织器
│   │   │   │   ├── FormatConverter.py # 格式转换器
│   │   │   │   └── MetadataWriter.py  # 元数据写入
│   │   │   ├── formats/               # 各种输出格式
│   │   │   │   ├── gaussian_splatting.py # GS 格式
│   │   │   │   ├── nerf_format.py     # NeRF 格式
│   │   │   │   └── colmap_format.py   # COLMAP 格式
│   │   │   └── index.py
│   │   └── README.md
│   │
│   ├── pipeline-orchestrator/         # 流程编排模块
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── Pipeline.py        # 主流程
│   │   │   │   ├── StageExecutor.py   # 阶段执行器
│   │   │   │   └── DependencyManager.py # 依赖管理
│   │   │   ├── workflows/             # 预定义工作流
│   │   │   │   ├── full_pipeline.yaml
│   │   │   │   └── quick_preview.yaml
│   │   │   └── index.py
│   │   └── README.md
│   │
│   └── shared/                        # 共享模块
│       ├── types/                     # 共享类型
│       │   ├── FrameData.py
│       │   ├── CameraParams.py
│       │   └── ProcessingResult.py
│       ├── utils/                     # 共享工具
│       │   ├── file_utils.py
│       │   ├── image_utils.py
│       │   └── logger.py
│       ├── constants/                 # 共享常量
│       │   └── formats.py
│       └── interfaces/                # 共享接口
│           ├── IProcessor.py         # 处理器接口
│           └── IValidator.py         # 验证器接口
│
├── core/                              # 核心基础设施
│   ├── config/
│   │   ├── global_config.yaml        # 全局配置
│   │   └── pipeline_config.yaml      # 流程配置
│   ├── database/
│   │   ├── cache_manager.py          # 缓存管理（避免重复处理）
│   │   └── metadata_db.py            # 元数据数据库
│   ├── logger/
│   │   └── pipeline_logger.py        # 流程日志
│   └── monitoring/
│       ├── progress_tracker.py       # 进度跟踪
│       └── performance_monitor.py    # 性能监控
│
├── cli/                               # 命令行接口
│   ├── main.py                        # 主入口
│   ├── commands/
│   │   ├── process.py                # 处理命令
│   │   ├── validate.py               # 验证命令
│   │   └── preview.py                # 预览命令
│   └── interactive.py                # 交互式界面
│
├── scripts/                           # 工具脚本
│   ├── generate-module.py            # 模块生成器
│   ├── batch-process.py              # 批处理脚本
│   └── benchmark.py                  # 性能测试
│
├── configs/                           # 配置文件目录
│   ├── presets/                      # 预设配置
│   │   ├── default.yaml
│   │   ├── high-quality.yaml
│   │   └── fast-preview.yaml
│   └── templates/                    # 配置模板
│
├── output/                            # 输出目录（被 .gitignore）
│   ├── images/                       # 提取的图片
│   ├── cameras/                      # 相机参数
│   ├── metadata/                     # 元数据
│   └── reports/                      # 验证报告
│
├── tests/                             # 集成测试
│   ├── integration/
│   │   └── test_full_pipeline.py
│   ├── e2e/
│   │   └── test_end_to_end.py
│   └── fixtures/                     # 测试数据
│       └── sample_video.mp4
│
├── docs/                              # 文档
│   ├── architecture.md               # 架构设计
│   ├── pipeline-stages.md            # 各阶段说明
│   ├── configuration-guide.md        # 配置指南
│   └── api-reference.md              # API 参考
│
├── requirements.txt                   # Python 依赖
├── pyproject.toml                    # 项目配置
├── setup.py                          # 安装脚本
└── README.md                         # 项目说明
```


### 3. 命名规范

#### 文件和目录命名

| 类型 | 规范 | 示例 | 使用场景 |
|------|------|------|----------|
| 模块目录 | `kebab-case` | `user-management`, `payment-gateway` | 所有模块目录 |
| 源文件 | `kebab-case` | `user-service.ts`, `auth-middleware.py` | 一般源代码文件 |
| 类文件 | `PascalCase` | `UserModel.ts`, `PaymentProcessor.py` | 包含单个类的文件 |
| 测试文件 | `*.test.*` 或 `*.spec.*` | `user-service.test.ts`, `auth.spec.py` | 测试文件 |
| 配置文件 | `kebab-case.ext` | `database-config.json`, `app-settings.yaml` | 配置文件 |
| 脚本文件 | `kebab-case` | `generate-module.py`, `run-tests.sh` | 可执行脚本 |

#### 代码命名

| 类型 | 规范 | 示例 | 说明 |
|------|------|------|------|
| 类名 | `PascalCase` | `UserService`, `PaymentGateway` | 所有类和接口 |
| 函数/方法 | `camelCase` | `getUserById()`, `processPayment()` | 函数和方法名 |
| 变量 | `camelCase` | `userName`, `totalAmount` | 局部变量和参数 |
| 常量 | `UPPER_SNAKE_CASE` | `MAX_RETRY_COUNT`, `API_BASE_URL` | 全局常量 |
| 私有成员 | `_camelCase` | `_internalCache`, `_processData()` | 私有属性和方法 |
| 接口 | `I` + `PascalCase` | `IUserRepository`, `IPaymentService` | 接口定义 |
| 类型别名 | `PascalCase` + `Type` | `UserDataType`, `ConfigOptionsType` | TypeScript 类型 |

### 4. 4DGS 数据处理模块设计模式

#### 4.1 处理器接口（IProcessor）

所有处理模块都应实现统一的处理器接口，方便流程编排：

```python
# modules/shared/interfaces/IProcessor.py
#WDD [2026-01-19] [4DGS 数据处理统一接口]

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..types.ProcessingResult import ProcessingResult

class IProcessor(ABC):
    """处理器基类接口"""
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否有效"""
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any], config: Optional[Dict] = None) -> ProcessingResult:
        """执行处理逻辑"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """返回此模块依赖的其他模块名称"""

// 导出配置
export { userModuleConfig } from './config';

// 默认导出（可选）
import { UserService } from './core/UserService';
export default UserService;
```

#### 4.2 模块配置管理

```typescript
// modules/user-management/src/config/index.ts

export interface UserModuleConfig {
  maxUsersPerPage: number;
  passwordMinLength: number;
  sessionTimeout: number;
}

export const defaultConfig: UserModuleConfig = {
  maxUsersPerPage: 20,
  passwordMinLength: 8,
  sessionTimeout: 3600,
};

// 允许外部覆盖配置
let config: UserModuleConfig = { ...defaultConfig };

export function setUserModuleConfig(newConfig: Partial<UserModuleConfig>) {
  config = { ...config, ...newConfig };
}

export function getUserModuleConfig(): UserModuleConfig {
  return { ...config };
}
```

#### 4.3 模块间通信

使用依赖注入和事件总线模式：

```typescript
// modules/order-management/src/core/OrderService.ts

import { IPaymentService } from '../../../payment-gateway/src';
import { EventBus } from '../../../../core/events';

export class OrderService {
  constructor(
    private paymentService: IPaymentService,
    private eventBus: EventBus
  ) {}

  async createOrder(orderData: IOrderCreateDTO): Promise<IOrder> {
    // 创建订单逻辑
    const order = await this.repository.create(orderData);
    
    // 发布事件而不是直接调用其他模块
    this.eventBus.emit('order.created', { orderId: order.id });
    
    return order;
  }
}
```

### 5. 模块文档规范

每个模块必须包含 README.md：

```markdown
# 模块名称

## 概述
简要描述模块的功能和职责

## 依赖关系
- 依赖的其他模块
- 外部依赖包

## 公共 API
### UserService
- `getUserById(id: string): Promise<User>`
- `createUser(data: UserCreateDTO): Promise<User>`

## 配置选项
| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| maxUsersPerPage | number | 20 | 每页最大用户数 |

## 使用示例
\`\`\`typescript
import { UserService } from './modules/user-management';

const userService = new UserService(config);
const user = await userService.getUserById('123');
\`\`\`

## 注意事项
任何特殊的使用限制或注意事项
```

### 6. 模块生成脚本

创建一个脚本来快速生成新模块：

```python
# scripts/generate-module.py
import os
import sys
from pathlib import Path

def create_module(module_name: str, base_path: str = "modules"):
    """
    生成新模块的基础结构
    
    用法: python scripts/generate-module.py <module-name>
    """
    module_path = Path(base_path) / module_name
    
    # 创建目录结构
    directories = [
        module_path / "src" / "core",
        module_path / "src" / "api",
        module_path / "src" / "models",
        module_path / "src" / "utils",
        module_path / "tests" / "unit",
        module_path / "tests" / "integration",
        module_path / "docs",
        module_path / "config",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        # 创建 __init__.py 使其成为 Python 包
        (directory / "__init__.py").touch()
    
    # 创建基础文件
    create_index_file(module_path / "src" / "index.ts")
    create_readme(module_path / "README.md", module_name)
    create_config(module_path / "config" / "index.ts")
    
    print(f"✅ 模块 '{module_name}' 已创建在 {module_path}")

def create_index_file(path: Path):
    content = """// 模块入口文件
#WDD [2026-01-19] [自动生成的模块入口文件]

export * from './core';
export * from './models';
"""
    path.write_text(content)

def create_readme(path: Path, module_name: str):
    content = f"""# {module_name.replace('-', ' ').title()}

## 概述
描述此模块的功能和职责

## 依赖关系
- 列出依赖的其他模块

## 公共 API
描述导出的接口和函数

## 配置选项
列出可配置参数

## 使用示例
提供使用代码示例
"""
    path.write_text(content)

def create_config(path: Path):
    content = """// 模块配置
#WDD [2026-01-19] [自动生成的配置文件]

export interface ModuleConfig {
  // 添加配置项
}

export const defaultConfig: ModuleConfig = {
  // 默认配置值
};
"""
    path.write_text(content)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/generate-module.py <module-name>")
        sys.exit(1)
    
    create_module(sys.argv[1])
```

### 7. 最佳实践检查清单

在开发模块时，确保：

#### 结构方面
- [ ] 模块在独立目录下
- [ ] 有清晰的 src/tests/docs 分离
- [ ] 包含 README.md 文档
- [ ] 有明确的入口文件 (index.ts/py)

#### 代码方面
- [ ] 遵循命名规范
- [ ] 接口和实现分离
- [ ] 避免循环依赖
- [ ] 使用依赖注入而非硬编码依赖

#### 文档方面
- [ ] 每个模块有 README
- [ ] 公共 API 有清晰的文档
- [ ] 配置选项有说明
- [ ] 提供使用示例

#### 测试方面
- [ ] 单元测试覆盖核心逻辑
- [ ] 集成测试验证模块间交互
- [ ] 测试文件位置规范

### 8. 4DGS 数据处理管道工作流程

#### 8.1 快速开始流程

```bash
# 1. 使用生成脚本创建核心处理模块
python scripts/generate-module.py video-input --lang python
python scripts/generate-module.py frame-extraction --lang python
python scripts/generate-module.py camera-estimation --lang python

# 2. 创建流程编排器
python scripts/generate-module.py pipeline-orchestrator --lang python

# 3. 运行数据处理
python cli/main.py process --input video.mp4 --output ./output --preset default
```

#### 8.2 模块开发工作流

**阶段 1: 规划处理流程**
   ```bash
   # 使用生成脚本
   python scripts/generate-module.py user-management
   python scripts/generate-module.py payment-gateway
   ```

4. **实现模块**
   - 从模块的核心业务逻辑开始
   - 定义清晰的公共接口
   - 实现配置管理
   - 编写单元测试

5. **集成模块**
   - 通过依赖注入连接模块
   - 使用事件总线处理模块间通信
   - 编写集成测试

6. **文档化**
   - 更新模块 README
   - 更新项目架构文档
   - 记录配置选项和 API

## 🚀 典型的 4DGS 数据处理流程示例

### 完整处理流程代码示例

```python
# cli/main.py
#WDD [2026-01-19] [4DGS 数据处理主流程]

import argparse
from modules.pipeline_orchestrator.src.core.Pipeline import Pipeline
from core.config.global_config import load_config
from core.logger.pipeline_logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='4DGS 数据预处理管道')
│   └── shared/                 # 共享代码
│
├── core/                       # 核心基础设施
│   ├── database/
│   ├── cache/
│   ├── logger/
│   └── events/
│
├── services/                   # 通用服务
│   ├── email/
│   ├── sms/
│   └── analytics/
│
├── api/                        # API 层
│   ├── rest/                   # REST API
│   ├── graphql/                # GraphQL API
│   └── websocket/              # WebSocket
│
├── scripts/
│   ├── generate-module.py
│   ├── build.sh
│   └── test.sh
│
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── deployment.md
│
└── README.md
```

## 工具推荐

- **Monorepo 管理**: Nx, Turborepo, Lerna
- **依赖注入**: InversifyJS (TS), dependency-injector (Python)
- **事件总线**: EventEmitter3, PyPubSub
- **文档生成**: TypeDoc, Sphinx
- **代码检查**: ESLint, Pylint (配置模块命名规范)

---

## 快速开始

1. 复制此 skill 中的目录结构模板
2. 运行 `generate-module.py` 创建新模块
3. 按照最佳实践检查清单开发模块
4. 使用依赖注入和事件总线连接模块
5. 编写文档和测试

#WDD [2026-01-19] [创建模块化架构 skill]
