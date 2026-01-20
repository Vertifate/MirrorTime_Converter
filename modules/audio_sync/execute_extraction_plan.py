import os
import json
import argparse
import shutil
import cv2
import sys
import torch
import torch.multiprocessing as mp
from queue import Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ffmpeg

# 确保可以导入同一目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def worker_inference_process(gpu_id, checkpoint_dir, task_queue, result_queue, device_str, cpu_threads): # <--- [新增参数 cpu_threads]
    """
    Step 4 的子进程工作函数 - 真正延迟导入版
    """
    try:
        # [关键修改] 1. 设置环境变量，严格限制 OpenMP 和 MKL 的线程数
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

        # 1. [最优先] 设置环境变量隔离显卡
        # 此时还没有 import EMAVFIPredictor，所以 torch.cuda 绝对没有被初始化
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            effective_device_str = "cuda:0" # 隔离后，物理卡 N 变成了逻辑卡 0
        else:
            effective_device_str = "cpu"

        # 2. [安全检查] 确认隔离成功
        # 此时 torch 应该只能看到 1 张卡
        import torch # 确保 torch 已导入
        if gpu_id >= 0:
            if torch.cuda.device_count() != 1:
                raise RuntimeError(f"致命错误：显卡隔离失败！Worker {os.getpid()} 应该只看到 1 张卡，但看到了 {torch.cuda.device_count()} 张。")

        # [关键修改] 2. 再次通过 API 强制限制线程 (双重保险)
        import cv2
        cv2.setNumThreads(cpu_threads)
        torch.set_num_threads(cpu_threads)

        # 3. [延迟导入] 现在环境已经安全了，再导入重型模型库
        # 将 import 移到这里！
        try:
            # 必须把这一行从文件顶层移到这里
            from ema_vfi_predictor import EMAVFIPredictor 
        except ImportError as e:
             result_queue.put({'status': 'fatal', 'msg': f"ImportError inside worker: {e}"})
             return

        # 4. 初始化模型
        # 注意：使用 effective_device_str ("cuda:0")
        model = EMAVFIPredictor(checkpoint_dir=checkpoint_dir, device=effective_device_str)
        
        print(f"[Worker-{os.getpid()}] Ready on {effective_device_str} (Physical: {gpu_id})")

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                break
            
            img0_path, img1_path, output_path, timestep, vid_name = task
            
            try:
                if not os.path.exists(img0_path) or not os.path.exists(img1_path):
                    result_queue.put({'status': 'error', 'msg': f"Missing input files for {vid_name}"})
                    continue

                img0 = cv2.imread(img0_path)
                img1 = cv2.imread(img1_path)
                
                if img0 is None or img1 is None:
                    result_queue.put({'status': 'error', 'msg': f"Read failed for {vid_name}"})
                    continue

                # 推理
                result_img = model.predict(img0, img1, timestep)
                
                # 保存
                cv2.imwrite(output_path, result_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(output_path, result_img)
                
                result_queue.put({'status': 'ok'})
                
            except RuntimeError as re:
                if 'out of memory' in str(re):
                    torch.cuda.empty_cache()
                    result_queue.put({'status': 'error', 'msg': f"OOM on {vid_name} (GPU {gpu_id}): {re}"})
                else:
                    result_queue.put({'status': 'error', 'msg': f"RuntimeError on {vid_name}: {re}"})
            except Exception as e:
                result_queue.put({'status': 'error', 'msg': f"Error on {vid_name}: {e}"})

    except Exception as e:
        import traceback
        result_queue.put({'status': 'fatal', 'msg': f"Worker crashed: {e}\n{traceback.format_exc()}"})

class ExtractionExecutor:
    def __init__(self, plan_data, video_dir, output_dir, cache_dir, checkpoint_dir, 
                 output_structure='by_frame', start_frame=None, end_frame=None, 
                 device='cuda', resolution_scale=1.0, workers_per_gpu=2, cpu_threads=1):
        
        self.full_plan = plan_data
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_structure = output_structure
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.resolution_scale = resolution_scale
        self.workers_per_gpu = workers_per_gpu
        self.cpu_threads = cpu_threads
        
        if device == 'cpu':
            self.num_gpus = 0
            self.device_list = ['cpu']
        else:
            if torch.cuda.is_available():
                self.num_gpus = torch.cuda.device_count()
                self.device_list = [f'cuda:{i}' for i in range(self.num_gpus)]
                print(f"\n[System] Detected {self.num_gpus} GPUs. Activating multi-GPU parallel mode.")
            else:
                print("\n[System] No GPU detected. Falling back to CPU mode.")
                self.num_gpus = 0
                self.device_list = ['cpu']

        self.filtered_plan = {}
        
    def analyze_requirements(self):
        """Step 1: 筛选计划范围"""
        print("\n" + "="*20 + " Step 1: Analyzing & Filtering " + "="*20)
        # ... (保持原有逻辑不变)
        sorted_keys = sorted(self.full_plan.keys())
        for key in sorted_keys:
            try:
                frame_idx = int(key.split('_')[1])
            except (IndexError, ValueError):
                continue
            if self.start_frame is not None and frame_idx < self.start_frame: continue
            if self.end_frame is not None and frame_idx > self.end_frame: continue
            self.filtered_plan[key] = self.full_plan[key]
        print(f"Total frames in plan: {len(self.full_plan)}")
        print(f"Frames to process: {len(self.filtered_plan)}")

    def populate_cache(self):
        """Step 2: 并行提取源帧"""
        # ... (保持原有逻辑不变，此处省略以节省空间，直接使用之前提供的 populate_cache 和 _ffmpeg_extract_worker 代码即可)
        print("\n" + "="*20 + " Step 2: Populating Cache (Multi-GPU FFmpeg) " + "="*20)
        os.makedirs(self.cache_dir, exist_ok=True)
        video_map = {}
        for root, dirs, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith(('.mov', '.mp4', '.avi', '.m4v')):
                    video_map[f] = os.path.join(root, f)
        
        video_indices_needed = {} 
        for frame_key, frame_data in self.filtered_plan.items():
            for vid_name, action_data in frame_data['videos'].items():
                if self.output_structure == 'by_frame':
                    out_path = os.path.join(self.output_dir, frame_key, f"{os.path.splitext(vid_name)[0]}.png")
                else: 
                    out_path = os.path.join(self.output_dir, os.path.splitext(vid_name)[0], f"{frame_key}.png")
                if os.path.exists(out_path): continue

                if vid_name not in video_indices_needed: video_indices_needed[vid_name] = set()
                action = action_data.get('action')
                if action == 'extract':
                    video_indices_needed[vid_name].add(action_data['frame_idx'])
                elif action == 'interpolate':
                    video_indices_needed[vid_name].add(action_data['prev_frame_idx'])
                    video_indices_needed[vid_name].add(action_data['next_frame_idx'])

        tasks = []
        for i, (vid_name, indices) in enumerate(video_indices_needed.items()):
            if not indices: continue
            if vid_name not in video_map: continue
            # FFmpeg 提取任务不需要非常严格的 GPU 绑定，轮询即可
            assigned_gpu = i % self.num_gpus if self.num_gpus > 0 else None
            tasks.append({'vid_name': vid_name, 'full_path': video_map[vid_name], 'indices': sorted(list(indices)), 'gpu_id': assigned_gpu})

        if not tasks:
            print("所有所需帧已缓存，跳过提取。")
            return

        max_workers = (self.num_gpus * 4) if self.num_gpus > 0 else os.cpu_count() # FFmpeg 可以多开一点
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._ffmpeg_extract_worker, t) for t in tasks]
            for _ in tqdm(as_completed(futures), total=len(tasks), desc="[Step 2] Extracting"):
                pass

    def _ffmpeg_extract_worker(self, task):
         # ... (保持之前提供的逻辑)
         vid_name = task['vid_name']
         full_path = task['full_path']
         indices = task['indices']
         video_cache_dir = os.path.join(self.cache_dir, vid_name)
         os.makedirs(video_cache_dir, exist_ok=True)
         final_indices = [idx for idx in indices if not os.path.exists(os.path.join(video_cache_dir, f"frame_{idx:06d}.png"))]
         if not final_indices: return
         
         chunk_size = 50
         for i in range(0, len(final_indices), chunk_size):
            chunk = final_indices[i:i+chunk_size]
            select_expr = "+".join([f"eq(n,{idx})" for idx in chunk])
            temp_pattern = os.path.join(video_cache_dir, "temp_%04d.png")
            try:
                (ffmpeg.input(full_path).filter('select', select_expr)
                 .output(temp_pattern, vsync=0, start_number=0).overwrite_output().run(quiet=True))
                for j, true_idx in enumerate(chunk):
                    src = temp_pattern % j
                    dst = os.path.join(video_cache_dir, f"frame_{true_idx:06d}.png")
                    if os.path.exists(src): os.rename(src, dst)
            except Exception as e:
                print(f"[FFmpeg Error] {vid_name}: {e}")

    def execute_processing(self):
        """Step 3 & 4: 多GPU推理"""
        print("\n" + "="*20 + " Step 3: Preparing Tasks " + "="*20)
        
        inference_tasks = [] 
        copy_tasks = []      
        
        sorted_frames = sorted(self.filtered_plan.keys())
        for frame_key in sorted_frames:
            frame_data = self.filtered_plan[frame_key]
            for vid_name, action_data in frame_data['videos'].items():
                if self.output_structure == 'by_frame':
                    out_path = os.path.join(self.output_dir, frame_key, f"{os.path.splitext(vid_name)[0]}.png")
                else: 
                    out_path = os.path.join(self.output_dir, os.path.splitext(vid_name)[0], f"{frame_key}.png")
                
                if os.path.exists(out_path): continue
                
                video_cache_dir = os.path.join(self.cache_dir, vid_name)
                action = action_data.get('action')
                
                if action == 'extract':
                    idx = action_data.get('frame_idx')
                    src_path = os.path.join(video_cache_dir, f"frame_{idx:06d}.png")
                    copy_tasks.append((src_path, out_path))
                elif action == 'interpolate':
                    idx_prev = action_data.get('prev_frame_idx')
                    idx_next = action_data.get('next_frame_idx')
                    step = action_data['interp_step']
                    src_prev = os.path.join(video_cache_dir, f"frame_{idx_prev:06d}.png")
                    src_next = os.path.join(video_cache_dir, f"frame_{idx_next:06d}.png")
                    inference_tasks.append((src_prev, src_next, out_path, step, vid_name))

        print(f"Tasks: Inference (GPU)={len(inference_tasks)}, Copy (IO)={len(copy_tasks)}")
        
        if copy_tasks:
            print("Executing Copy Tasks...")
            for src, dst in tqdm(copy_tasks, desc="Copying Frames"):
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        
        if not inference_tasks:
            return

        print("\n" + "="*20 + " Step 4: Multi-GPU Inference " + "="*20)
        
        # [优化] 使用 spawn context 的原生 Queue，而不是 Manager().Queue()，速度更快
        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        
        for t in inference_tasks:
            task_queue.put(t)
            
        processes = []
        
        # [优化] 计算总进程数：GPU数量 * 每卡Worker数
        if self.num_gpus > 0:
            total_workers = self.num_gpus * self.workers_per_gpu
            print(f"Spawning {total_workers} workers ({self.workers_per_gpu} per GPU) on CUDA...")
        else:
            total_workers = max(1, os.cpu_count() // 2)
            print(f"Spawning {total_workers} workers on CPU...")

        for i in range(total_workers):
            if self.num_gpus > 0:
                # 轮询分配: Worker 0->GPU0, Worker 1->GPU1, Worker 2->GPU0 ...
                gpu_index = i % self.num_gpus
                dev_str = f"cuda:{gpu_index}"
                gpu_id = gpu_index
            else:
                dev_str = "cpu"
                gpu_id = -1
                
            p = ctx.Process(
                target=worker_inference_process,
                args=(gpu_id, self.checkpoint_dir, task_queue, result_queue, dev_str, self.cpu_threads) # <--- [传递参数到子进程]
            )
            p.start()
            processes.append(p)
            
        total_inf = len(inference_tasks)
        with tqdm(total=total_inf, desc="[Step 4] AI Interpolating") as pbar:
            completed = 0
            while completed < total_inf:
                try:
                    res = result_queue.get(timeout=2.0)
                    if res['status'] == 'ok':
                        pbar.update(1)
                        completed += 1
                    elif res['status'] == 'fatal':
                        print(f"\n[FATAL] {res['msg']}")
                        break
                    else:
                        print(f"\n[Error] {res['msg']}")
                        pbar.update(1)
                        completed += 1
                except Empty:
                    if not any(p.is_alive() for p in processes):
                        print("All workers died.")
                        break
        
        for p in processes:
            p.join()
        
        if self.device_list[0].startswith('cuda'):
            torch.cuda.empty_cache()

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("plan_json", help="Path to extraction_plan.json")
    parser.add_argument("video_dir", help="Original video dir")
    parser.add_argument("output_dir", help="Output dir")
    parser.add_argument("--checkpoint_dir", default="./InterpAny-Clearer/checkpoints/EMA-VFI/DR-EMA-VFI/train_sdi_log")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure", default='by_frame')
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="Number of workers per GPU (default: 2)")
    
    args = parser.parse_args()

    # ... (文件加载逻辑同前)
    with open(args.plan_json, 'r') as f:
        plan_data = json.load(f)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(args.output_dir, "cache")

    executor = ExtractionExecutor(
        plan_data=plan_data,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        cache_dir=cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_structure=args.structure,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        device=args.device,
        resolution_scale=args.scale,
        workers_per_gpu=args.workers_per_gpu # 传递参数
    )
    
    executor.analyze_requirements()
    executor.populate_cache()
    executor.execute_processing()

if __name__ == "__main__":
    main()