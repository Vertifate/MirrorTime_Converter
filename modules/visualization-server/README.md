# MirrorTime Converter 使用指南

## 快速启动

### 方式一：使用 launch.py（推荐）

一键启动整个系统：

```bash
python launch.py
```

这个脚本会自动：
1. 检查并安装依赖
2. 启动后端服务器（端口 8000）
3. 启动前端界面（端口 5173）
4. 打开浏览器访问 http://localhost:5173

按 `Ctrl+C` 退出所有服务。

### 方式二：手动启动

**启动后端：**
```bash
cd modules/visualization-server/src
python main.py
```

**启动前端：**
```bash
cd visualization-ui
npm run dev
```

## 功能说明

### 可视化界面

访问 http://localhost:5173 查看实时监控界面：

- **顶部状态栏**：显示连接状态
- **控制面板**：启动/停止处理按钮
- **处理阶段面板**：显示 7 个处理阶段的实时进度

### 处理阶段

1. **视频输入** - 视频验证和元数据提取
2. **帧提取** - 智能帧采样
3. **图像预处理** - 图像增强
4. **相机参数估计** - COLMAP/SfM
5. **位姿精化** - Bundle Adjustment
6. **数据验证** - 质量检查
7. **输出格式化** - 生成训练数据

### API 端点

- `http://localhost:8000/docs` - API 文档
- `http://localhost:8000/api/pipeline/status` - 获取流程状态
- `http://localhost:8000/api/pipeline/start` - 启动流程
- `http://localhost:8000/api/pipeline/stop` - 停止流程
- `ws://localhost:8000/ws` - WebSocket 实时通信

## 开发说明

### 目录结构

```
MirrorTime_Converter/
├── launch.py                      # 主启动脚本
├── visualization-ui/              # 前端界面
│   ├── src/
│   │   ├── App.tsx               # 主应用组件
│   │   └── App.css               # Dark 主题样式
│   └── package.json
└── modules/
    └── visualization-server/      # 后端服务器
        ├── src/
        │   └── main.py           # FastAPI 服务器
        └── requirements.txt
```

### 添加新的处理模块

编辑 `modules/visualization-server/src/main.py`，在 `pipeline_state.stages` 中添加新阶段。

### 自定义主题

编辑 `visualization-ui/src/App.css` 中的 CSS 变量。

#WDD [2026-01-19] [可视化模块使用文档]
