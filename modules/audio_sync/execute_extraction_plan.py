import os
import json
import argparse
import shutil
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ffmpeg
from PIL import Image, PngImagePlugin

# 确保可以导入同一目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def _get_raw_timestamps(video_path):
    """
    使用 ffprobe 快速获取视频每一帧的原始 PTS 时间戳。
    """
    try:
        probe = ffmpeg.probe(
            video_path,
            select_streams='v:0',
            show_entries='frame=pkt_pts_time'
        )
        timestamps = [float(frame['pkt_pts_time']) for frame in probe.get('frames', []) if 'pkt_pts_time' in frame]
        return np.array(timestamps)
    except ffmpeg.Error as e:
        print(f"[FFmpeg 错误] 无法读取 {os.path.basename(video_path)}: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        print(f"[未知错误] 在处理 {os.path.basename(video_path)} 时发生: {e}")
        return None

class SimpleExtractor:
    def __init__(self, snapped_data, video_dir, output_dir, workers=os.cpu_count(), 
                 start_frame=None, end_frame=None, output_structure='by_frame', output_format='jpg'):
        """
        简化版提取器：直接根据 snapped_data 提取帧。
        """
        self.snapped_data = snapped_data
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.workers = workers
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output_structure = output_structure
        self.output_format = output_format.lower()
        
        # 缓存每个视频的完整时间戳，用于将时间转换为索引
        self.video_full_timestamps = {}

    def _preload_timestamps(self, video_paths):
        """并行获取所需视频的完整时间戳"""
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_path = {executor.submit(_get_raw_timestamps, path): path for path in video_paths}
            
            with tqdm(total=len(future_to_path), desc="[准备] 扫描视频帧索引") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        self.video_full_timestamps[path] = future.result()
                    except Exception as e:
                        print(f"Error process {path}: {e}")
                    pbar.update(1)

    def _ffmpeg_extract_worker(self, task):
        """
        FFmpeg 提取单个视频的一组帧，并写入元数据。
        task: { 'vid_name': ..., 'full_path': ..., 'indices': [ (frame_idx, output_path, meta_dict), ... ] }
        """
        vid_name = task['vid_name']
        full_path = task['full_path']
        indices = task['indices']
        
        if not indices: return

        # 分块处理以避免命令行过长
        chunk_size = 50
        for i in range(0, len(indices), chunk_size):
            chunk = indices[i:i+chunk_size] # List of (frame_idx, output_abs_path, meta_dict)
            
            # 构造 select 表达式: eq(n,idx1)+eq(n,idx2)...
            select_expr = "+".join([f"eq(n,{idx})" for idx, _, _ in chunk])
            
            # 使用 temp pattern 输出
            temp_dir = os.path.join(self.output_dir, "temp_extract", vid_name)
            os.makedirs(temp_dir, exist_ok=True)
            
            ext = f".{self.output_format}"
            temp_pattern = os.path.join(temp_dir, f"chunk_{i}_%04d{ext}")
            
            try:
                # 运行 FFmpeg
                # For JPG, we might want to set quality using qscale:v or q:v. Defaulting to high quality.
                ffmpeg_args = {'vsync': 0, 'start_number': 0}
                if self.output_format in ['jpg', 'jpeg']:
                     ffmpeg_args['q:v'] = 2 # High quality for JPG
                
                (ffmpeg.input(full_path)
                 .filter('select', select_expr)
                 .output(temp_pattern, **ffmpeg_args)
                 .overwrite_output()
                 .run(quiet=True))
                 
                # 将临时文件重命名/移动到最终目标路径
                # 注意：对于 JPG，我们暂时不注入复杂的元数据（原代码用的是 PngInfo）
                # 如果需要元数据，建议保存为 sidecar json 或写入 EXIF (较复杂)
                for j, (true_idx, target_path, meta) in enumerate(chunk):
                    src = temp_pattern % j
                    if os.path.exists(src):
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.move(src, target_path)

                        # --- Embed Metadata (UserComment) ---
                        try:
                            # 1. 打开图片
                            with Image.open(target_path) as img:
                                # 2. 准备 metadata string (JSON)
                                # meta 已经在上层循环中准备好了，这里直接用
                                # 添加额外的 drift/offset 信息，如果 meta 中没有的话，
                                # 但 meta 目前来自于 task['indices'] -> (idx, path, meta_dict)
                                # task['indices'] 的 meta 已经在 execute() 中构建完整了
                                
                                meta_json = json.dumps(meta)
                                
                                # 3. 获取或创建 EXIF 数据
                                exif = img.getexif()
                                # 0x9286 = UserComment
                                exif[0x9286] = meta_json
                                
                                # 4. 保存回去 (原地覆盖)
                                # 注意: 使用 quality='keep' 或指定高质量，以免重压缩损失画质
                                # q:v=2 ~ 95 quality roughly
                                img.save(target_path, exif=exif, quality=95)
                                
                        except Exception as ex_meta:
                            print(f"[Metadata Error] Failed to embed info for {os.path.basename(target_path)}: {ex_meta}")
                                
            except Exception as e:
                print(f"[FFmpeg Error] {vid_name}: {e}")
        
        # 清理临时目录
        shutil.rmtree(os.path.join(self.output_dir, "temp_extract", vid_name), ignore_errors=True)

    def execute(self):
        print("\n" + "="*20 + " 开始执行帧提取 (含元数据) " + "="*20)
        
        # 1. 收集需要提取的任务
        tasks_by_video = {}
        
        # 遍历 snapped_data (按视频组织的)
        for vid_name, data in self.snapped_data.items():
            file_path = data.get('file_path')
            mapping = data.get('mapping', [])
            
            if not file_path or not mapping: continue
            
            for i, item in enumerate(mapping):
                # 过滤帧范围
                if self.start_frame is not None and i < self.start_frame: continue
                if self.end_frame is not None and i > self.end_frame: continue
                
                # 计算输出路径
                frame_key = f"frame_{i:06d}"
                ext = f".{self.output_format}"
                if self.output_structure == 'by_frame':
                    out_path = os.path.join(self.output_dir, frame_key, f"{os.path.splitext(vid_name)[0]}{ext}")
                else:
                    out_path = os.path.join(self.output_dir, os.path.splitext(vid_name)[0], f"{frame_key}{ext}")
                
                # 如果已存在则跳过
                if os.path.exists(out_path): continue
                
                if file_path not in tasks_by_video:
                    tasks_by_video[file_path] = []
                
                # 获取对齐参数
                offset = data.get('offset_seconds', 0.0)
                drift = data.get('drift_scale', 1.0)
                
                # 准备元数据所需信息
                ideal_time = item['ideal_time']
                snapped_time = item['snapped_time']
                
                # 计算最终的全局同步时间 (Real Global Time)
                # 公式: Global = Local * Drift + Offset
                global_time = snapped_time * drift + offset
                
                time_error = snapped_time - ideal_time
                
                meta_info = {
                    'ideal_time': ideal_time,
                    'real_time': snapped_time,
                    'global_time': global_time,
                    'time_error': time_error
                }
                
                # 存储暂时只能用 time 来找 index
                tasks_by_video[file_path].append( { 
                    'time': snapped_time, 
                    'out_path': out_path,
                    'meta': meta_info,
                    'frame_idx': item.get('frame_idx')
                } )

        if not tasks_by_video:
            print("所有帧已存在或无任务。")
            return

        # 2. 检查索引完整性并决定是否需要扫描
        has_indices = all(t.get('frame_idx') is not None for tasks in tasks_by_video.values() for t in tasks)
        
        if not has_indices:
            print("准备视频时间轴数据 (未发现缓存索引)...")
            self._preload_timestamps(list(tasks_by_video.keys()))
        else:
            print("[优化] 发现缓存的帧索引，跳过时间轴扫描。")

        # 3. 转换时间为索引，并构建最终任务列表
        final_tasks = []
        
        for file_path, items in tasks_by_video.items():
            vid_name = os.path.basename(file_path)
            
            indices_list = []
            
            # 分支路径：有索引直接用，无索引扫描找最近
            if has_indices:
                for item in items:
                     indices_list.append( (item['frame_idx'], item['out_path'], item['meta']) )
            else:
                timestamps = self.video_full_timestamps.get(file_path)
                if timestamps is None:
                    print(f"跳过 {vid_name} (无法读取元数据)")
                    continue
                    
                for item in items:
                    t = item['time']
                    # 寻找最近的索引
                    idx = (np.abs(timestamps - t)).argmin()
                    indices_list.append( (idx, item['out_path'], item['meta']) )
            
            if not indices_list: continue
            
            # 排序索引
            indices_list.sort(key=lambda x: x[0])
            
            final_tasks.append({
                'vid_name': vid_name,
                'full_path': file_path,
                'indices': indices_list
            })

        # 4. 并行执行提取
        print(f"提交 {len(final_tasks)} 个视频提取任务...")
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self._ffmpeg_extract_worker, t) for t in final_tasks]
            for _ in tqdm(as_completed(futures), total=len(final_tasks), desc="[提取] Processing Videos"):
                pass
        
        # 清理
        temp_root = os.path.join(self.output_dir, "temp_extract")
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)
            
        print("提取完成。")

def main():
    pass 

if __name__ == "__main__":
    main()