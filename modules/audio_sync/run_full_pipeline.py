import os
import argparse
import shutil
import sys
import time
import json
import multiprocessing as mp
from PIL import Image
import numpy as np

# 确保可以导入同一目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_sync import AudioSyncSystem
from execute_extraction_plan import CacheExtractor, _get_raw_timestamps

class FullSyncPipeline:
    def __init__(self, 
                 video_dir, 
                 output_dir, 
                 chirp_duration=0.3,
                 start_freq=2000,
                 end_freq=6000,
                 sample_rate=48000,
                 matching_window=3.0,
                 start_frame=0,
                 end_frame=1,
                 output_structure='by_frame',
                 workers=4,
                 batch_size=10,
                 buffer_seconds=1.0):
        """
        全流程同步管线 (Cache First -> Sync -> Finalize).
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        
        # 音频同步参数
        self.chirp_duration = chirp_duration
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.sample_rate = sample_rate
        self.matching_window = matching_window
        
        # 提取参数
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output_structure = output_structure
        self.workers = workers
        self.batch_size = batch_size
        self.buffer_seconds = buffer_seconds

        # 内部路径
        self.cache_dir = os.path.join(self.output_dir, "cache")
        # 结果文件，用于记录同步信息
        self.sync_info_path = os.path.join(self.output_dir, "sync_results.json")

    def log(self, message, header=False):
        """统一的日志输出格式"""
        if header:
            print("\n" + "="*40)
            print(f" {message}")
            print("="*40)
        else:
            print(f"[Run] {message}")

    def _scan_videos(self):
        """扫描视频文件"""
        video_files = []
        for root, dirs, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith(('.mov', '.mp4', '.avi', '.m4v')):
                    video_files.append(os.path.join(root, f))
        return sorted(video_files)

    def _phase_1_cache_extraction(self, video_files):
        """
        第一阶段：全量（区间+Buffer）预提取
        (Smart Cache Check integrated in CacheExtractor)
        """
        self.log("Phase 1: 预提取帧到缓存 (Buffered Extraction)", header=True)
        
        # 我们依靠 CacheExtractor 内部的快速检查来判断是否跳过
        print("[Smart Cache] 正在检查缓存文件完整性...")
        
        extractor = CacheExtractor(
            video_dir=self.video_dir,
            cache_dir=self.cache_dir,
            workers=self.workers,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            buffer_seconds=self.buffer_seconds
        )
        extractor.run()

    def _phase_2_audio_sync(self, video_files):
        """
        第二阶段：音频同步分析
        """
        self.log("Phase 2: 音频同步分析", header=True)
        
        syncer = AudioSyncSystem(
            chirp_duration=self.chirp_duration,
            start_freq=self.start_freq,
            end_freq=self.end_freq,
            sample_rate=self.sample_rate
        )
        
        alignment_results = syncer.align_videos(
            video_files, 
            matching_window_seconds=self.matching_window, 
            visualize=False, 
            tqdm_desc="[Sync] Analyzing Audio"
        )
        
        if not alignment_results:
            self.log("未检测到有效同步信号。将执行无同步直出流程。")
            return None
            
        with open(self.sync_info_path, 'w') as f:
            json.dump(alignment_results, f, indent=4)
            
        return alignment_results

    def _get_fast_duration(self, video_path):
        """快速获取视频时长（无需扫描全文件）"""
        import subprocess
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except Exception:
            # Fallback (slow but reliable)
            ts = _get_raw_timestamps(video_path)
            return ts[-1] if ts else 0.0

    def _process_single_target_frame(self, i, ref_fps, intersect_start, video_info, video_files, fallback_extractor):
        """Helper for parallel processing of a single target frame"""
        # 相对时间
        rel_time = i / ref_fps
        # 绝对全局时间
        abs_global_time = intersect_start + rel_time
        
        frame_dir_name = f"frame_{i:06d}"
        
        for v_path in video_files:
            fname = os.path.basename(v_path)
            v_data = video_info[fname]
            
            # 逆推本地时间
            if v_data['is_synced']:
                local_time = (abs_global_time - v_data['offset']) / v_data['drift']
            else:
                local_time = rel_time 
            
            # 计算本地帧号
            local_frame_idx = int(round(local_time * v_data['fps']))
            
            if local_frame_idx < 0:
                continue
                
            # 寻找缓存
            vid_stem = os.path.splitext(fname)[0]
            cache_filename = f"{vid_stem}_frame_{local_frame_idx:06d}.jpg"
            cache_path = os.path.join(self.cache_dir, cache_filename)
            
            # 输出路径
            if self.output_structure == 'by_frame':
                dst_dir = os.path.join(self.output_dir, frame_dir_name)
                dst_name = f"{vid_stem}.jpg"
            else:
                dst_dir = os.path.join(self.output_dir, vid_stem)
                dst_name = f"{frame_dir_name}.jpg"
            
            dst_path = os.path.join(dst_dir, dst_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Metadata
            meta_to_write = {
                "MirrorTime": rel_time
            }
            
            if os.path.exists(cache_path):
                # HIT
                try:
                    shutil.copy2(cache_path, dst_path)
                    self._inject_metadata(dst_path, meta_to_write)
                except Exception:
                    pass
            else:
                # MISS
                fallback_extractor.extract_single_frame_fallback(
                    v_path, local_frame_idx, dst_path, meta_to_write
                )

    def _phase_3_finalize(self, video_files, sync_results):
        """
        第三阶段：生成最终输出 (Intersection & Shift)
        """
        self.log("Phase 3: 计算公共区域并生成输出", header=True)
        
        # 1. 构造 video info map
        video_info = {}
        syncer = AudioSyncSystem()
        
        sycned_map = {}
        if sync_results:
            for res in sync_results:
                fname = os.path.basename(res['file'])
                sycned_map[fname] = res
        
        self.log("计算视频同步参数与公共区域...")
        
        from tqdm import tqdm
        for v_path in tqdm(video_files, desc="[Sync] Calc Parameters"):
            fname = os.path.basename(v_path)
            fps = syncer._get_video_fps(v_path)
            if fps is None: fps = 30.0
            
            # 优化：使用 Fast Probe 替代全量扫描
            duration = self._get_fast_duration(v_path)
            
            info = {
                'path': v_path,
                'fps': fps,
                'duration': duration,
                'offset': 0.0,
                'drift': 1.0,
                'is_synced': False
            }
            
            # 填入同步参数
            # Global = Local * Drift + Offset
            # Local Start = 0 -> Global Start = Offset
            # Local End = Duration -> Global End = Duration * Drift + Offset
            if fname in sycned_map:
                info['offset'] = sycned_map[fname].get('offset_seconds', 0.0)
                info['drift'] = sycned_map[fname].get('drift_scale', 1.0)
                info['is_synced'] = True
                
            info['global_start'] = info['offset']
            info['global_end'] = (info['duration'] * info['drift']) + info['offset']
            
            video_info[fname] = info

        # 2. 计算 Intersection (公共区域)
        # Intersection Start = Max(All Starts)
        # Intersection End = Min(All Ends)
        
        all_starts = [v['global_start'] for v in video_info.values()]
        all_ends = [v['global_end'] for v in video_info.values()]
        
        if not all_starts:
            self.log("无有效视频信息，退出。")
            return

        intersect_start = max(all_starts)
        intersect_end = min(all_ends)
        intersect_duration = intersect_end - intersect_start
        
        print("\n" + "-"*40)
        print(f" [同步分析报告]")
        print(f" 公共区域起点 (Global 0): {intersect_start:.4f} s")
        print(f" 公共区域终点:           {intersect_end:.4f} s")
        print(f" 公共区域时长:           {intersect_duration:.4f} s")
        print("-" * 40 + "\n")

        if intersect_duration <= 0:
            self.log("错误：视频之间没有公共重叠区域！无法同步。")
            return

        # 3. 确定参考FPS
        ref_fps = 30.0
        if video_files:
            first_name = os.path.basename(video_files[0])
            ref_fps = video_info[first_name]['fps']
            
        # 4. 确定最终输出范围
        # Global Time Origin shift to Intersect Start
        # Output Time T corresponds to Real Global Time (Intersect_Start + T)
        
        max_valid_frames = int(intersect_duration * ref_fps)
        
        # 用户指定的 end_frame 是相对于 New Origin 的
        # 所以我们需要限制 end_frame
        
        output_end_frame = self.end_frame
        if output_end_frame > max_valid_frames:
            print(f"[警告] 用户请求结束帧 {self.end_frame} 超出公共区域最大帧数 {max_valid_frames}。")
            print(f"       已自动截断至帧 {max_valid_frames}。")
            output_end_frame = max_valid_frames
            
        self.log(f"最终输出范围: {self.start_frame} - {output_end_frame} (FPS: {ref_fps:.2f})")

        # 实例化 Fallback Extractor (Workers=1 is fine as it's called inside threads)
        fallback_extractor = CacheExtractor(
            self.video_dir, self.cache_dir, workers=1
        )
        
        # Parallel Execution
        # We use ThreadPoolExecutor to parallelize IO operations (Copy, Inject, Fallback Call)
        # Note: Fallback uses subprocess which is robust.
        
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        
        target_frames = list(range(self.start_frame, output_end_frame + 1))
        
        self.log(f"开始生成帧 (多线程并发: {self.workers} workers)...")
        
        # Define the task function with fixed arguments
        process_func = partial(self._process_single_target_frame, 
                               ref_fps=ref_fps, 
                               intersect_start=intersect_start, 
                               video_info=video_info, 
                               video_files=video_files, 
                               fallback_extractor=fallback_extractor)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            list(tqdm(executor.map(process_func, target_frames), total=len(target_frames), desc="[Finalize] Processing"))

    def _inject_metadata(self, image_path, meta_dict):
        """写入 EXIF UserComment"""
        try:
            meta_json = json.dumps(meta_dict)
            with Image.open(image_path) as img:
                exif = img.getexif()
                exif[0x9286] = meta_json
                img.save(image_path, exif=exif, quality=95)
        except Exception:
            pass

    def _cleanup(self):
        """清理缓存"""
        if os.path.exists(self.cache_dir):
            self.log(f"清理缓存目录: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)

    def run(self):
        self.log("启动抽帧流程 (Cache-First)", header=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 0. 扫描
        video_files = self._scan_videos()
        if not video_files:
            self.log("无视频文件。")
            return

        # 1. 预提取
        # 如果缓存目录已有东西，可能需要询问用户或由 CacheExtractor 内部判断跳过
        self._phase_1_cache_extraction(video_files)
        
        # 2. 同步
        sync_results = self._phase_2_audio_sync(video_files)
        
        # 3. 输出
        self._phase_3_finalize(video_files, sync_results)
        
        # 4. 清理
        self._cleanup()
        
        self.log("全流程结束。", header=True)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="多机位同步工具 (V2 Cache First)")
    
    parser.add_argument("video_dir", help="包含原始视频的目录")
    parser.add_argument("output_dir", help="结果输出目录")
    
    parser.add_argument("--start_frame", type=int, default=0, help="目标起始帧")
    parser.add_argument("--end_frame", type=int, default=10, help="目标结束帧")
    parser.add_argument("--structure", choices=['by_frame', 'by_video'], default='by_frame', help="输出结构")
    parser.add_argument("--workers", type=int, default=4, help="线程数")
    parser.add_argument("--buffer", type=float, default=0, help="预提取缓冲时长(秒)")
    # 保留部分 Sync 参数
    parser.add_argument("--window", type=float, default=3.0, help="Sync Window")
    
    args = parser.parse_args()

    pipeline = FullSyncPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        output_structure=args.structure,
        workers=args.workers,
        buffer_seconds=args.buffer,
        matching_window=args.window
    )
    
    pipeline.run()
