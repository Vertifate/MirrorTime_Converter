import os
import argparse
import shutil
import sys
import time
import json
import torch.multiprocessing as mp

# 确保可以导入同一目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_sync import AudioSyncSystem
from snap_frames import FrameSnapper
from execute_extraction_plan import SimpleExtractor

class FullSyncPipeline:
    def __init__(self, 
                 video_dir, 
                 output_dir, 
                 chirp_duration=0.3,
                 start_freq=2000,
                 end_freq=6000,
                 sample_rate=48000,
                 matching_window=3.0,
                 start_frame=None,
                 end_frame=None,
                 output_structure='by_frame',
                 workers=4,
                 resolution_scale=1.0,
                 batch_size=10):
        """
        初始化全流程同步管线。
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
        self.resolution_scale = resolution_scale
        self.batch_size = batch_size

        # 内部路径
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.snap_cache_path = os.path.join(self.video_dir, "snapped_frames_cache.json")

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

    def _load_cache(self):
        """尝试读取缓存"""
        if os.path.exists(self.snap_cache_path):
            try:
                with open(self.snap_cache_path, 'r') as f:
                    data = json.load(f)
                self.log(f"已加载提取计划: {self.snap_cache_path}")
                return data
            except Exception as e:
                self.log(f"读取计划失败: {e}，将重新运行。", header=True)
        return None

    def _save_cache(self, data):
        """保存全局缓存"""
        try:
            with open(self.snap_cache_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"[Plan] 计划已更新至: {self.snap_cache_path}")
        except Exception as e:
            print(f"[Plan] 保存失败: {e}")

    def _run_global_alignment(self, video_files):
        """步骤 1: 全局音频同步"""
        print("\n" + "-"*30)
        print(" [步骤 1] 音频同步分析 (全局)")
        print("-"*30)
        
        syncer = AudioSyncSystem(
            chirp_duration=self.chirp_duration,
            start_freq=self.start_freq,
            end_freq=self.end_freq,
            sample_rate=self.sample_rate
        )
        
        return syncer.align_videos(video_files, matching_window_seconds=self.matching_window, visualize=False, tqdm_desc="[同步] 分析视频音频")

    def _snap_batch(self, alignment_batch):
        """步骤 2: 批次帧映射"""
        snapper = FrameSnapper(
            alignment_batch, 
            max_workers=self.workers, 
            start_frame=self.start_frame, 
            end_frame=self.end_frame
        )
        return snapper.snap_all_videos()

    def generate_extraction_plan(self, video_files):
        """
        阶段 1: 快速制定抽帧计划 (Generate Extraction Plan)
        包括：音频同步 -> 批量扫描元数据并映射帧 -> 生成完整的 snapped_cache
        """
        self.log("阶段 1: 制定抽帧计划", header=True)
        
        # 1. 全局音频同步
        alignment_results = self._run_global_alignment(video_files)
        if not alignment_results:
            self.log("音频同步失败，流程终止。", header=True)
            return None
            
        # 2. 批量映射 (Snapping)
        # 即使只做 plan，我们也可以分批 snap 避免内存压力，但主要是为了快速完成
        total_items = len(alignment_results)
        num_batches = (total_items + self.batch_size - 1) // self.batch_size
        
        global_snapped_data = {}
        
        print("\n" + "-"*30)
        print(" [步骤 2] 快速映射真实帧 (制定计划)")
        print("-"*30)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_items)
            current_batch = alignment_results[start_idx:end_idx]
            
            print(f" >> [Plan Batch {i+1}/{num_batches}] 扫描视频 {start_idx+1}-{end_idx}...")
            
            batch_snapped = self._snap_batch(current_batch)
            if batch_snapped:
                global_snapped_data.update(batch_snapped)
                
                # 每批次保存一次，防止长时间运行丢失
                self._save_cache(global_snapped_data)
        
        self.log(f"计划制定完成！共包含 {len(global_snapped_data)} 个视频的提取任务。")
        return global_snapped_data

    def run_extraction_batches(self, full_snapped_data):
        """
        阶段 2: 执行抽帧 (Execute Extraction)
        读取完整的 plan，分批执行提取。
        """
        self.log("阶段 2: 执行抽帧任务", header=True)
        
        items_to_process = list(full_snapped_data.items())
        items_to_process.sort(key=lambda x: x[0])
        
        total_items = len(items_to_process)
        num_batches = (total_items + self.batch_size - 1) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_items)
            
            # current_batch 是 [(key, val), ...] 形式
            current_batch_list = items_to_process[start_idx:end_idx]
            # 转回 dict 供 extractor 使用
            current_batch_dict = dict(current_batch_list)
            
            print("\n" + "-"*30)
            print(f" [提取批次 {i+1}/{num_batches}] 处理视频 {start_idx+1}-{end_idx}")
            print("-"*30)
            
            executor = SimpleExtractor(
                snapped_data=current_batch_dict,
                video_dir=self.video_dir,
                output_dir=self.output_dir,
                workers=self.workers,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                output_structure=self.output_structure,
                output_format='jpg'
            )
            executor.execute()

    def run(self):
        self.log("启动视频同步与提取管线", header=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 0. 扫描资源
        video_files = self._scan_videos()
        if not video_files:
            self.log(f"错误: 在 {self.video_dir} 中未找到视频文件。", header=True)
            return

        # 1. 检查是否存在计划
        global_snapped_data = self._load_cache()
        
        if global_snapped_data is None:
            # 如果没有计划，执行阶段 1
            global_snapped_data = self.generate_extraction_plan(video_files)
        
        if global_snapped_data is None:
            self.log("无法生成提取计划，任务终止。")
            return

        # 2. 执行阶段 2
        self.run_extraction_batches(global_snapped_data)

        # 完成
        self.log(f"全流程完成！结果已保存至: {self.output_dir}", header=True)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="一键式多机位视频同步与帧提取工具")
    
    parser.add_argument("video_dir", help="包含原始视频的目录")
    parser.add_argument("output_dir", help="结果输出目录")
    
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧 (可选)")
    parser.add_argument("--end_frame", type=int, default=0, help="结束帧 (可选)")
    parser.add_argument("--structure", choices=['by_frame', 'by_video'], default='by_frame', help="输出目录结构")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数 (建议设置为 CPU 核心数的 1/4 或更少，避免死机)")
    parser.add_argument("--window", type=float, default=1.0, help="同步匹配窗口大小(秒)")
    parser.add_argument("--batch_size", type=int, default=20, help="批处理大小 (每次处理多少个视频)")

    args = parser.parse_args()

    pipeline = FullSyncPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        matching_window=args.window,
        output_structure=args.structure,
        workers=args.workers,
        batch_size=args.batch_size
    )
    
    pipeline.run()
