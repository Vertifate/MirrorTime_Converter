import os
import argparse
import shutil
import sys
import time
import json

# 确保可以导入同一目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from audio_sync import AudioSyncSystem
from snap_frames import FrameSnapper
from plan_extraction import ExtractionPlanner
from execute_extraction_plan import ExtractionExecutor
import torch.multiprocessing as mp

class FullSyncPipeline:
    def __init__(self, 
                 video_dir, 
                 output_dir, 
                 checkpoint_dir,
                 chirp_duration=0.3,
                 start_freq=2000,
                 end_freq=6000,
                 sample_rate=48000,
                 matching_window=3.0,
                 start_frame=None,
                 end_frame=None,
                 output_structure='by_frame',
                 device='cuda',
                 workers=os.cpu_count(),
                 resolution_scale=1.0,
                 workers_per_gpu=1,
                 cpu_threads=1):
        """
        初始化全流程同步管线。
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        
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
        self.device = device
        self.workers = workers
        self.resolution_scale = resolution_scale
        self.workers_per_gpu = workers_per_gpu
        self.cpu_threads = cpu_threads

        # 内部路径
        self.cache_dir = os.path.join(self.output_dir, "cache")


    def run(self):
        print("\n" + "="*40)
        print(" 启动全流程视频同步与帧提取管线")
        print("="*40)

        # 0. 准备工作
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 扫描视频文件
        video_files = []
        for root, dirs, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith(('.mov', '.mp4', '.avi', '.m4v')):
                    video_files.append(os.path.join(root, f))
        
        if not video_files:
            print(f"错误: 在 {self.video_dir} 中未找到视频文件。")
            return
        
        # 检查是否存在缓存的提取计划
        plan_cache_path = os.path.join(self.video_dir, "extraction_plan_cache.json")
        extraction_plan = None
        
        if os.path.exists(plan_cache_path):
            print(f"\n[提示] 检测到缓存的提取计划: {plan_cache_path}")
            print("       将跳过前三步 (同步、映射、规划)，直接使用缓存计划。")
            try:
                with open(plan_cache_path, 'r') as f:
                    extraction_plan = json.load(f)
            except Exception as e:
                print(f"[警告] 读取缓存计划失败: {e}。将重新执行完整流程。")
                extraction_plan = None

        if extraction_plan is None:
            start_time = time.time()
            
            # 1. 音频同步 (Audio Sync)
            print("\n" + "-"*30)
            print(" [步骤 1/4] 音频同步分析")
            print("-"*30)
            
            syncer = AudioSyncSystem(
                chirp_duration=self.chirp_duration,
                start_freq=self.start_freq,
                end_freq=self.end_freq,
                sample_rate=self.sample_rate
            )
            
            # 注意：align_videos 需要视频文件列表
            alignment_data = syncer.align_videos(video_files, matching_window_seconds=self.matching_window, visualize=False, tqdm_desc="[同步] 分析视频音频")
            
            if not alignment_data:
                print("错误: 音频同步失败，无法继续。")
                return

            # 2. 真实帧映射 (Frame Snapping)
            print("\n" + "-"*30)
            print(" [步骤 2/4] 映射真实帧时间")
            print("-"*30)
            
            snapper = FrameSnapper(alignment_data, max_workers=self.workers)
            snapped_data = snapper.snap_all_videos()
            
            if not snapped_data:
                print("错误: 帧映射失败，无法继续。")
                return

            # 3. 制定提取计划 (Plan Extraction)
            print("\n" + "-"*30)
            print(" [步骤 3/4] 制定提取与插帧计划")
            print("-"*30)
            
            planner = ExtractionPlanner(snapped_data, max_workers=self.workers)
            extraction_plan = planner.plan()
            
            if not extraction_plan:
                print("错误: 计划制定失败，无法继续。")
                return
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n[计时] 前三步总计用时: {elapsed_time:.2f} 秒")
            
            # 保存计划到缓存
            try:
                with open(plan_cache_path, 'w') as f:
                    json.dump(extraction_plan, f, indent=4)
                print(f"[缓存] 提取计划已保存至: {plan_cache_path}")
            except Exception as e:
                print(f"[警告] 无法保存缓存计划: {e}")

        # 4. 执行提取与插帧 (Execute Plan)
        print("\n" + "-"*30)
        print(" [步骤 4/4] 执行提取与 AI 插帧")
        print("-"*30)
        
        step4_start_time = time.time()

        executor = ExtractionExecutor(
            plan_data=extraction_plan,
            video_dir=self.video_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            checkpoint_dir=self.checkpoint_dir,
            output_structure=self.output_structure,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            device=self.device,
            resolution_scale=self.resolution_scale,
            workers_per_gpu=self.workers_per_gpu,
            cpu_threads=self.cpu_threads
        )
        
        executor.analyze_requirements()
        executor.populate_cache()
        executor.execute_processing()
        
        step4_end_time = time.time()
        elapsed_step4 = step4_end_time - step4_start_time
        print(f"\n[计时] 第四步总计用时: {elapsed_step4:.2f} 秒")
        
        # --- 新增：统计并输出插帧比例 ---
        if executor.filtered_plan:
            total_actions = 0
            extract_count = 0
            interp_count = 0
            
            plan_to_analyze = executor.filtered_plan
            
            for frame_data in plan_to_analyze.values():
                for vid_data in frame_data['videos'].values():
                    total_actions += 1
                    if vid_data.get('action') == 'extract':
                        extract_count += 1
                    elif vid_data.get('action') == 'interpolate':
                        interp_count += 1
            
            if total_actions > 0:
                print("\n" + "="*20 + " 任务摘要 " + "="*20)
                print(f"处理的总帧数: {len(plan_to_analyze)}")
                print(f"  - 直接抽帧 (Extract): {extract_count} ({extract_count/total_actions*100:.1f}%)")
                print(f"  - AI 插帧 (Interpolate): {interp_count} ({interp_count/total_actions*100:.1f}%)")

        # 可选：清理缓存
        # if os.path.exists(self.cache_dir):
        #     shutil.rmtree(self.cache_dir)
        
        print("\n" + "="*40)
        print(f" 全流程完成！结果已保存至: {self.output_dir}")
        print("="*40)

if __name__ == "__main__":

    # ====================================================================
    # --- 关键修正：强制设置多进程启动方式为 spawn ---
    # ====================================================================
    try:
        # 必须在所有其他多进程代码执行前设置
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # ====================================================================
    # --- 参数配置区 (PARAMETER CONFIGURATION) ---
    # ====================================================================
    parser = argparse.ArgumentParser(description="一键式多机位视频同步与帧提取工具")
    
    parser.add_argument("video_dir", help="包含原始视频的目录")
    parser.add_argument("output_dir", help="结果输出目录")
    parser.add_argument("--checkpoint_dir", default="./InterpAny-Clearer/checkpoints/EMA-VFI/DR-EMA-VFI/train_sdi_log", help="EMA-VFI 模型权重目录")
    
    parser.add_argument("--start_frame", type=int, default=None, help="起始帧 (可选)")
    parser.add_argument("--end_frame", type=int, default=None, help="结束帧 (可选)")
    parser.add_argument("--structure", choices=['by_frame', 'by_video'], default='by_frame', help="输出目录结构")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="并行线程数")
    parser.add_argument("--scale", type=float, default=0.5, help="输出图像的分辨率缩放比例 (例如 0.5)")
    parser.add_argument("--window", type=float, default=1.0, help="同步匹配窗口大小(秒)")
    parser.add_argument("--workers_per_gpu", type=int, default=3, help="AI 插帧阶段每个 GPU 承载的并行进程数 (默认: 2)")
    parser.add_argument("--cpu_threads", type=int, default=1, help="每个推理子进程允许使用的 CPU 线程数 (建议: 1)") # <--- [新增命令行参数]

    args = parser.parse_args()

    pipeline = FullSyncPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        matching_window=args.window,
        output_structure=args.structure,
        workers=args.workers,
        resolution_scale=args.scale,
        workers_per_gpu=args.workers_per_gpu,
        cpu_threads=args.cpu_threads
    )
    
    pipeline.run()
