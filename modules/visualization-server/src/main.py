"""
MirrorTime Converter 可视化服务器
FastAPI 后端，提供 REST API 和 WebSocket 实时通信
"""
#WDD [2026-01-19] [创建可视化服务器后端]

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime

app = FastAPI(
    title="MirrorTime Converter API",
    description="4DGS 数据预处理可视化监控服务",
    version="1.0.0"
)

# CORS 配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """向所有连接的客户端广播消息"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Send failed: {e}")

manager = ConnectionManager()

# 流程状态存储
pipeline_state = {
    "current_stage": None,
    "stages": [
        {
            "id": "video-input",
            "name": "视频输入",
            "status": "pending",  # pending, running, completed, failed
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "frame-extraction",
            "name": "帧提取",
            "status": "pending",
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "image-preprocessing",
            "name": "图像预处理",
            "status": "pending",
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "camera-estimation",
            "name": "相机参数估计",
            "status": "pending",
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "pose-refinement",
            "name": "位姿精化",
            "status": "pending",
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "data-validation",
            "name": "数据验证",
            "status": "pending",
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "output-formatter",
            "name": "输出格式化",
            "status": "pending",
            "progress": 0,
            "message": "",
            "start_time": None,
            "end_time": None,
        },
    ]
}

# ============ REST API 路由 ============

@app.get("/")
async def root():
    return {
        "name": "MirrorTime Converter API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """获取流程状态"""
    return pipeline_state

@app.post("/api/pipeline/start")
async def start_pipeline(config: Dict[str, Any]):
    """Start data processing pipeline"""
    # #WDD [2026-01-19] [Accept project directory parameter]
    project_dir = config.get("project_directory", "")
    print(f"Starting pipeline")
    print(f"  Project Directory: {project_dir}")
    print(f"  Config: {config}")
    
    # Reset all stage statuses
    for stage in pipeline_state["stages"]:
        stage["status"] = "pending"
        stage["progress"] = 0
        stage["message"] = ""
        stage["start_time"] = None
        stage["end_time"] = None
    
    pipeline_state["current_stage"] = "video-input"
    
    # Broadcast status update
    await manager.broadcast({
        "type": "pipeline_started",
        "data": pipeline_state
    })
    
    # Execute processing pipeline asynchronously
    asyncio.create_task(run_pipeline(config))
    
    return {"status": "started", "message": "Pipeline started"}

@app.post("/api/pipeline/stop")
async def stop_pipeline():
    """Stop pipeline"""
    print("Stopping pipeline")
    
    await manager.broadcast({
        "type": "pipeline_stopped",
        "data": pipeline_state
    })
    
    return {"status": "stopped"}

@app.get("/api/stages")
async def get_stages():
    """获取所有处理阶段信息"""
    return {"stages": pipeline_state["stages"]}

@app.get("/api/stages/{stage_id}")
async def get_stage_detail(stage_id: str):
    """获取特定阶段的详细信息"""
    for stage in pipeline_state["stages"]:
        if stage["id"] == stage_id:
            return stage
    return {"error": "Stage not found"}, 404

# ============ WebSocket 路由 ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接端点，用于实时推送流程状态"""
    await manager.connect(websocket)
    
    try:
        # 发送初始状态
        await websocket.send_json({
            "type": "initial_state",
            "data": pipeline_state
        })
        
        # 保持连接并接收客户端消息
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理客户端请求
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "request_status":
                await websocket.send_json({
                    "type": "status_update",
                    "data": pipeline_state
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============ 处理流程模拟 ============

async def run_pipeline(config: Dict[str, Any]):
    """
    Execute data processing pipeline
    This is a simulation - should call real processing modules in production
    """
    print("Executing pipeline...")
    
    for stage in pipeline_state["stages"]:
        stage_id = stage["id"]
        
        # 更新当前阶段
        pipeline_state["current_stage"] = stage_id
        stage["status"] = "running"
        stage["start_time"] = datetime.now().isoformat()
        stage["message"] = f"正在执行 {stage['name']}..."
        
        # 广播状态更新
        await manager.broadcast({
            "type": "stage_started",
            "data": {
                "stage_id": stage_id,
                "stage": stage,
                "pipeline": pipeline_state
            }
        })
        
        # 模拟处理进度
        for progress in range(0, 101, 10):
            stage["progress"] = progress
            stage["message"] = f"{stage['name']} 进度: {progress}%"
            
            await manager.broadcast({
                "type": "progress_update",
                "data": {
                    "stage_id": stage_id,
                    "progress": progress,
                    "message": stage["message"]
                }
            })
            
            await asyncio.sleep(0.5)  # 模拟处理时间
        
        # 完成当前阶段
        stage["status"] = "completed"
        stage["progress"] = 100
        stage["end_time"] = datetime.now().isoformat()
        stage["message"] = f"{stage['name']} 已完成"
        
        await manager.broadcast({
            "type": "stage_completed",
            "data": {
                "stage_id": stage_id,
                "stage": stage
            }
        })
        
        await asyncio.sleep(0.5)  # 阶段间隔
    
    # 流程完成
    pipeline_state["current_stage"] = None
    
    await manager.broadcast({
        "type": "pipeline_completed",
        "data": pipeline_state
    })
    
    print("Pipeline completed")

# ============ 启动服务器 ============

if __name__ == "__main__":
    import uvicorn
    
    print("Starting MirrorTime Converter visualization server...")
    print("WebSocket: ws://localhost:8000/ws")
    print("API: http://localhost:8000/api")
    print("Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
