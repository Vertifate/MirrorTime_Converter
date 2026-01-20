import os
import sys
import asyncio
import json
import base64
import io
import threading
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

try:
    import cv2
    from PIL import Image
except ImportError:
    print("[!] OpenCV or Pillow not found. Please install: pip install opencv-python pillow")
    sys.exit(1)

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# #WDD [2026-01-20] [架构终极锁定：移除 tk 回退，物理隔离项目数据，修复 UI 循环闪烁]

app = FastAPI(title="MirrorTime Converter GUI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 状态管理
class State:
    def __init__(self):
        self.pipeline = {
            "current_stage": None,
            "stages": [
                {"id": "video_to_frames", "name": "Videos", "status": "pending", "progress": 0, "message": ""},
                {"id": "colmap_sfm", "name": "COLMAP SfM", "status": "pending", "progress": 0, "message": ""},
                {"id": "colmap_undistort", "name": "Image Undistortion", "status": "pending", "progress": 0, "message": ""},
                {"id": "segmentation", "name": "Dynamic Mask Segmentation", "status": "pending", "progress": 0, "message": ""},
                {"id": "format_export", "name": "Export 4DGS Format", "status": "pending", "progress": 0, "message": ""},
            ],
            "videos": []
        }
        self.connections: List[WebSocket] = []

    async def broadcast(self, msg_type: str, data: Any):
        content = {"type": msg_type, "data": data}
        for ws in self.connections:
            try:
                await ws.send_json(content)
            except:
                pass

state = State()

def get_video_info(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
        ret, frame = cap.read()
        cap.release()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
        return {
            "filename": os.path.basename(path),
            "path": path,
            "size_mb": round(os.path.getsize(path)/(1024*1024), 2),
            "duration": round(frames/fps, 2) if fps>0 else 0,
            "frames": frames,
            "resolution": f"{w}x{h}",
            "raw_frame": frame_rgb
        }
    except: return None

def get_cache_path(project_dir: str) -> Path:
    return Path(project_dir) / ".mirrortime" / "video_cache.json"

async def load_project_cache(project_dir: str):
    """从项目目录加载缓存 (不广播状态 update 避免循环)"""
    if not project_dir or not os.path.isdir(project_dir): return False
    cache_file = get_cache_path(project_dir)
    if not cache_file.exists(): return False
    try:
        # 挂载缩略图
        thumb_dir = Path(project_dir) / ".mirrortime" / "thumbnails"
        if thumb_dir.exists():
            app.mount("/thumb", StaticFiles(directory=str(thumb_dir)), name="thumb")
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
            state.pipeline["videos"] = data.get("videos", [])
            state.pipeline["stages"][0].update({"status": "completed", "progress": 100, "message": f"Loaded {len(state.pipeline['videos'])} videos from project cache."})
            print(f"[*] Loaded cached project data from: {project_dir}")
            return True
    except: return False

async def run_video_scan(project_dir: str, force_refresh: bool = False):
    """扫描处理逻辑"""
    if not project_dir or not os.path.isdir(project_dir): return
    
    # 强制刷新前重置
    if force_refresh:
        state.pipeline["videos"] = []
    
    # 1. 尝试静默加载
    if not force_refresh:
        if await load_project_cache(project_dir):
            await state.broadcast("status_update", state.pipeline)
            return

    # 2. 完整扫描流程
    v_stage = state.pipeline["stages"][0]
    v_stage.update({"status": "running", "progress": 0, "message": "Scanning project filesystem..."})
    await state.broadcast("status_update", state.pipeline)

    videos = []
    exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    found = []
    for p in [project_dir, os.path.join(project_dir, "videos")]:
        if os.path.isdir(p):
            for f in os.listdir(p):
                if any(f.lower().endswith(e) for e in exts):
                    found.append(os.path.join(p, f))
    
    cache_dir = Path(project_dir) / ".mirrortime"
    thumb_dir = cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(found):
        info = get_video_info(path)
        if info:
            raw = info.pop("raw_frame", None)
            if raw is not None:
                img_name = f"{info['filename']}.jpg"
                Image.fromarray(raw).save(thumb_dir/img_name, quality=80)
                info["thumbnail"] = f"/thumb/{img_name}"
            videos.append(info)
            state.pipeline["videos"] = list(videos)
            v_stage.update({"progress": int((i+1)/len(found)*100), "message": f"Processing {i+1}/{len(found)}..."})
            await state.broadcast("status_update", state.pipeline)

    with open(get_cache_path(project_dir), 'w') as f:
        json.dump({"videos": videos, "updated_at": datetime.now().isoformat()}, f)
    
    app.mount("/thumb", StaticFiles(directory=str(thumb_dir)), name="thumb")
    v_stage.update({"status": "completed", "progress": 100, "message": f"Finalized {len(videos)} videos."})
    await state.broadcast("status_update", state.pipeline)

@app.get("/api/utils/select-directory")
def select_dir():
    # #WDD [2026-01-20] [仅使用 zenity，严禁使用 tk]
    try:
        proc = subprocess.Popen(['zenity', '--file-selection', '--directory', '--title=Project Root'], stdout=subprocess.PIPE, text=True)
        out, _ = proc.communicate()
        if proc.returncode == 0: return {"directory": out.strip()}
    except: pass
    return {"directory": ""}

@app.post("/api/pipeline/scan")
async def api_scan(data: Dict[str, Any]):
    asyncio.create_task(run_video_scan(data.get("project_directory"), data.get("force", False)))
    return {"ok": True}

@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    state.connections.append(websocket)
    try:
        await websocket.send_json({"type": "initial_state", "data": state.pipeline})
        while True: await websocket.receive_text()
    except WebSocketDisconnect: state.connections.remove(websocket)

# UI 静态挂载
ui_path = Path(__file__).parent / "static"
if ui_path.exists():
    if (ui_path/"assets").exists(): app.mount("/assets", StaticFiles(directory=str(ui_path/"assets")), name="assets")
    @app.get("/")
    async def index(): return FileResponse(ui_path/"index.html")

if __name__ == "__main__":
    def start_browse():
        import time; time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")
    threading.Thread(target=start_browse, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
