import os
import sys

# #WDD [2026-01-20] [Bugfix: 修复导入路径] 添加项目根目录到 sys.path，解决 ModuleNotFoundError: No module named 'modules'
current_dir = os.path.dirname(os.path.abspath(__file__))
# 修正：从 modules/gui 回溯到 MirrorTime_Converter 需要向上两级 (../..)
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
from fastapi.responses import FileResponse, Response

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
                {"id": "sync_frame_extraction", "name": "Sync Extraction", "status": "pending", "progress": 0, "message": ""},
                {"id": "colmap_sfm", "name": "COLMAP SfM", "status": "pending", "progress": 0, "message": ""},
                {"id": "segmentation", "name": "Dynamic Mask Segmentation", "status": "pending", "progress": 0, "message": ""},
                {"id": "format_export", "name": "Export 4DGS Format", "status": "pending", "progress": 0, "message": ""},
            ],
            "videos": [],
            # #WDD [2026-01-20] [新增] 控制台日志历史
            "logs": []
        }
        # #WDD [2026-01-20] [新增] 停止请求标志
        self.stop_requested = False
        self.connections: List[WebSocket] = []

    async def broadcast(self, msg_type: str, data: Any):
        content = {"type": msg_type, "data": data}
        for ws in self.connections:
            try:
                await ws.send_json(content)
            except:
                pass

state = State()

import concurrent.futures

def process_video_task(path, thumb_dir):
    """
    独立的处理函数，用于并行执行
    1. 抽取中间帧
    2. 缩放图片 (Height=240) #WDD [2026-01-20] 性能优化: 缩放+降低质量
    3. 保存为 JPG
    """
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release() # 获取完元数据直接释放，不进行耗时的 seek+read
        
        thumb_url = ""
        if thumb_dir:
            img_name = f"{os.path.basename(path)}.jpg"
            save_path = os.path.join(thumb_dir, img_name)
            
            # 使用 FFmpeg 进行极速抽取
            # -ss (input seeking) 是最快的定位方式
            # scale=512:-1 自动计算高度
            # -q:v 5 稍微降低质量换取速度
            mid_time = (frames / fps / 2) if fps > 0 else 0
            cmd = [
                'ffmpeg', 
                '-ss', f"{mid_time:.2f}",
                '-i', path,
                '-vf', 'scale=512:-2', # -2 保证偶数尺寸兼容性
                '-vframes', '1',
                '-q:v', '5',
                '-y',
                '-nostdin',       # 防止读取 stdin
                '-loglevel', 'error', # 减少日志输出
                save_path
            ]
            
            # 这里的 subprocess 调用虽然有开销，但相比 cv2 的软解seek通常要快得多
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0:
                thumb_url = f"/thumb/{img_name}"

        return {
            "filename": os.path.basename(path),
            "path": path,
            "size_mb": round(os.path.getsize(path)/(1024*1024), 2),
            "duration": round(frames/fps, 2) if fps>0 else 0,
            "frames": frames,
            "resolution": f"{w}x{h}",
            "thumbnail": thumb_url
        }
    except Exception as e:
        print(f"[!] Error processing {path}: {e}")
        return None

def get_cache_path(project_dir: str) -> Path:
    return Path(project_dir) / "mirrortime_results" / "video_cache.json"

async def load_project_cache(project_dir: str):
    """从项目目录加载缓存 (不广播状态 update 避免循环)"""
    if not project_dir or not os.path.isdir(project_dir): return False
    cache_file = get_cache_path(project_dir)
    if not cache_file.exists(): return False
    try:
        # 挂载缩略图
        thumb_dir = Path(project_dir) / "mirrortime_results" / "thumbnails"
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
    """扫描处理逻辑 - 并行化优化版"""
    if not project_dir or not os.path.isdir(project_dir): return
    
    if force_refresh:
        state.pipeline["videos"] = []
    else:
        if await load_project_cache(project_dir):
            await state.broadcast("status_update", state.pipeline)
            return

    v_stage = state.pipeline["stages"][0]
    v_stage.update({"status": "running", "progress": 0, "message": "Scanning project filesystem..."})
    await state.broadcast("status_update", state.pipeline)

    videos_found = []
    exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    for p in [project_dir, os.path.join(project_dir, "videos")]:
        if os.path.isdir(p):
            for f in os.listdir(p):
                if any(f.lower().endswith(e) for e in exts):
                    videos_found.append(os.path.join(p, f))
    
    cache_dir = Path(project_dir) / "mirrortime_results"
    thumb_dir = cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # 并行处理
    loop = asyncio.get_running_loop()
    # 根据 CPU 核心数决定并行度，保留一些余量
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    
    print(f"[*] Starting parallel scan with {max_workers} workers for {len(videos_found)} videos.")
    
    videos = []
    completed = 0
    total = len(videos_found)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            loop.run_in_executor(pool, process_video_task, path, str(thumb_dir)): path 
            for path in videos_found
        }
        
        # 收集结果
        for future in asyncio.as_completed(futures):
            result = await future
            completed += 1
            if result:
                videos.append(result)
            
            # 使用简单的节流: 每完成 5 个或者全部完成时更新一下 UI
            if completed % 5 == 0 or completed == total:
                # 排序保证列表稳定性 (可选，按文件名)
                # videos.sort(key=lambda x: x["filename"]) 
                state.pipeline["videos"] = videos # 弱一致性即可，这里不需要每次都深拷贝
                v_stage.update({"progress": int(completed/total*100), "message": f"Processing {completed}/{total}..."})
                await state.broadcast("status_update", state.pipeline)

    # 最终排序和保存
    videos.sort(key=lambda x: x["filename"])
    state.pipeline["videos"] = videos
    
    with open(get_cache_path(project_dir), 'w') as f:
        json.dump({"videos": videos, "updated_at": datetime.now().isoformat()}, f)
    
    app.mount("/thumb", StaticFiles(directory=str(thumb_dir)), name="thumb")
    v_stage.update({"status": "completed", "progress": 100, "message": f"Finalized {len(videos)} videos."})
    await state.broadcast("status_update", state.pipeline)

from modules.audio_sync.run_full_pipeline import FullSyncPipeline

class GUIStatusLogger:
    def __init__(self, callback):
        self.callback = callback
    def log_status(self, msg, progress=None):
        self.callback(msg, progress)

class GUISyncPipeline(FullSyncPipeline):
    """继承 Pipeline 并重写 log 方法以支持 WebSocket 进度推送"""
    def __init__(self, status_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_callback = status_callback
        
    def log(self, message, header=False):
        print(f"[GUI-Backend] {message}")
        # 尝试从消息中解析进度 (简单的启发式)
        # 实际上准确进度需要改造 backend，这里先做简单的文本反馈
        self.status_callback(message)

async def run_sync_extraction(project_dir: str, force_restart: bool = False):
    """
    同步拆帧逻辑实现 - 调用真实管线
    #WDD [2026-01-20] [重构: 集成 FullSyncPipeline, 支持断点续传/重来]
    """
    if not project_dir or not os.path.isdir(project_dir): return
    
    stage = next((s for s in state.pipeline["stages"] if s["id"] == "sync_frame_extraction"), None)
    if not stage: return

    state.pipeline["current_stage"] = stage["id"]
    status_msg = "Starting sync pipeline..."
    if force_restart:
        status_msg = "Force restarting sync pipeline..."
    
    stage.update({"status": "running", "progress": 0, "message": status_msg})
    await state.broadcast("status_update", state.pipeline)

    loop = asyncio.get_running_loop()

    def status_update(msg, progress=None):
        if progress is not None:
             stage["progress"] = progress
        stage["message"] = msg
        
        # #WDD [2026-01-20] [新增] 追加到日志历史（保留最近50条）
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {msg}"
        state.pipeline["logs"].append(log_entry)
        if len(state.pipeline["logs"]) > 50:
            state.pipeline["logs"] = state.pipeline["logs"][-50:]
        
        # #WDD [2026-01-20] [UI Feed] 修复进度不更新的问题: 使用 run_coroutine_threadsafe 跨线程推送
        asyncio.run_coroutine_threadsafe(state.broadcast("status_update", state.pipeline), loop)

    try:
        # 构造参数
        video_dir = project_dir # 假设根目录即视频目录，或者 videos 子目录? 
        # gui 代码之前的 scan 逻辑会看 project_dir 和 project_dir/videos
        # 这里为了稳妥，检查一下
        if os.path.join(project_dir, "videos"):
             src_video = os.path.join(project_dir, "videos") if os.path.exists(os.path.join(project_dir, "videos")) else project_dir
        
        # #WDD [2026-01-20] [User Req] 输出到根目录 input 文件夹
        output_dir = os.path.join(project_dir, "input")
        
        # 处理强制重开
        cache_file = os.path.join(src_video, "snapped_frames_cache.json")
        if force_restart and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"[*] Deleted cache plan: {cache_file}")
            except Exception as e:
                print(f"[!] Failed to delete cache: {e}")

        # 在线程中运行以避免阻塞 asyncio loop
        def _run_worker():
            pipeline = GUISyncPipeline(
                status_callback=status_update,
                video_dir=src_video,
                output_dir=output_dir,
                output_structure='by_frame',
                workers=max(1, (os.cpu_count() or 4) - 2), # 留点余地
                batch_size=10,
                # #WDD [2026-01-20] [UI Feed] 传递进度回调
                progress_callback=status_update,
                # #WDD [2026-01-20] [新增] 传递停止检查回调
                stop_check=lambda: state.stop_requested
            )
            pipeline.run()

        # 启动后台线程
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_worker)

        # 完成
        stage.update({
            "status": "completed", 
            "progress": 100, 
            "message": "Sync extraction pipeline finished."
        })
        state.pipeline["current_stage"] = None
        await state.broadcast("status_update", state.pipeline)

    except Exception as e:
        error_msg = str(e)
        if state.stop_requested:
            # #WDD [2026-01-20] [新增] 用户主动停止
            stage.update({"status": "stopped", "message": "Pipeline stopped by user. Progress saved."})
        else:
            stage.update({"status": "failed", "message": f"Error: {error_msg}"})
        state.pipeline["current_stage"] = None
        state.stop_requested = False  # Reset flag
        await state.broadcast("status_update", state.pipeline)
        print(f"[!] Sync Extraction stopped/failed: {e}")

@app.post("/api/pipeline/stop")
async def api_stop_pipeline():
    """
    #WDD [2026-01-20] [新增] 强制停止当前管线
    设置停止标志，管线会在下一个检查点安全停止
    """
    state.stop_requested = True
    stage = next((s for s in state.pipeline["stages"] if s["id"] == "sync_frame_extraction"), None)
    if stage and stage["status"] == "running":
        stage["message"] = "Stopping... (waiting for current task to finish)"
        await state.broadcast("status_update", state.pipeline)
    return {"status": "stop_requested"}

@app.post("/api/pipeline/visualize")
async def api_visualize(data: Dict[str, Any]):
    """获取可视化数据: 时间轴和多视角结构"""
    project_dir = data.get("project_dir")
    if not project_dir: return {"error": "No path"}
    
    # 尝试找到 cache
    # logic similar to run_sync
    src_video = project_dir
    possible = [project_dir, os.path.join(project_dir, "videos")]
    cache_path = None
    for p in possible:
        if os.path.exists(os.path.join(p, "snapped_frames_cache.json")):
            cache_path = os.path.join(p, "snapped_frames_cache.json")
            break
            
    if not cache_path:
        return {"error": "No synchronization plan found. Please run Sync Extraction first."}
        
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
        # Pivot data: Group by frame index/ideal time
        # 我们取第一个有数据的视频作为基准
        timeline = {}
        
        for vid, info in data.items():
            mapping = info.get("mapping", [])
            path = info.get("file_path")
            for item in mapping:
                idx = item.get("ideal_time") # Use ideal time as key (float) approx? No, let's use list index if consistent
                # 实际上 idx = item['frame_idx'] 是全局统一的 0, 1, 2...
                f_idx = item.get("frame_idx")
                
                if f_idx not in timeline:
                    timeline[f_idx] = {
                        "frame_idx": f_idx,
                        "ideal_time": item.get("ideal_time"),
                        "views": []
                    }
                
                timeline[f_idx]["views"].append({
                    "video": vid,
                    "path": path, # full path needs for thumb
                    "real_time": item.get("snapped_time"),
                    "global_time": item.get("snapped_time") * info.get("drift_scale", 1.0) + info.get("offset_seconds", 0.0)
                })
        
        # Convert to sorted list
        sorted_timeline = [timeline[k] for k in sorted(timeline.keys())]
        return {"timeline": sorted_timeline}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/pipeline/frame_thumb")
def get_frame_thumb(path: str, time: float):
    """获取指定视频在指定时间的缩略图"""
    if not os.path.exists(path): return FileResponse(ui_path/"assets/placeholder.png") # Fail gracefully
    
    try:
        # ffmpeg -ss time -i path -vframes 1 -f image2pipe -vcodec mjpeg -
        cmd = [
            'ffmpeg',
            '-ss', f"{time:.3f}",
            '-i', path,
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-vf', 'scale=320:-1', # 小图
            '-q:v', '5',
            '-nostdin',
            '-loglevel', 'error',
            '-' # Pipe output
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return Response(content=proc.stdout, media_type="image/jpeg")
    except: pass
    return Response(status_code=404)

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

@app.post("/api/pipeline/sync")
async def api_sync(data: Dict[str, Any]):
    asyncio.create_task(run_sync_extraction(data.get("project_directory"), data.get("force_restart", False)))
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
    from fastapi import Response # Missing import supplement
    def start_browse():
        import time; time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")
    threading.Thread(target=start_browse, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
