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
                 buffer_seconds=1.0,
                 skip_sync=False):
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
        self.skip_sync = skip_sync

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

    def _check_smart_cache(self):
        """
        检查是否跳过预提取。
        策略：只要 cache 目录存在且不为空，就认为缓存有效，直接跳过。
        """
        if not os.path.exists(self.cache_dir):
            return False
            
        # Optimization: Use scandir to find any jpg without listing all files
        try:
            with os.scandir(self.cache_dir) as it:
                for entry in it:
                    if entry.name.lower().endswith('.jpg') and entry.is_file():
                        print(f"[Smart Cache] 检测到缓存目录 {self.cache_dir} 包含图片文件，跳过预提取。")
                        return True
        except Exception:
            pass
        
        return False


    def _phase_1_audio_sync(self, video_files):
        """
        第一阶段：音频同步分析 (Audio Sync)
        """
        self.log("Phase 1: 音频同步分析", header=True)
        
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
            self.log("未检测到有效同步信号。将执行无同步直出流程 (Fallback defaults)。")
            return None
            
        # 2. 计算公共区域起始点 (Intersection Start) 并对齐
        # 我们需要所有视频的 Offset 和 Duration
        
        infos = []
        for res in alignment_results:
            v_path = res['file']
            offset = res['offset_seconds']
            drift = res['drift_scale']
            duration = self._get_fast_duration(v_path)
            
            g_start = offset
            g_end = offset + (duration * drift)
            infos.append({'res': res, 'start': g_start, 'end': g_end})
            
        # Find Global Zero Target
        # User Request: Global 0 = First Audio Chirp (Clapperboard)
        
        shift_target = 0.0
        shift_mode = "Unknown"
        
        # Try to find the first chirp time from reference
        # The 'alignment_results' usually contains 'matched_points' which has 'reference_peaks'
        # These peaks are in the Reference Timeline (which is the initial Global Timeline)
        
        ref_peaks = []
        for res in alignment_results:
             # Just grab the first non-empty reference peaks
             # Note: All results should theoretically have the same reference_peaks if they matched against the same ref
             peaks = res.get('matched_points', {}).get('reference_peaks', [])
             if peaks:
                 ref_peaks = peaks
                 break
        
        if ref_peaks:
             shift_target = ref_peaks[0]
             shift_mode = "First Chirp (Clapperboard)"
        else:
             # Fallback: Intersection Start
             all_starts = [i['start'] for i in infos]
             shift_target = max(all_starts) if all_starts else 0.0
             shift_mode = "Intersection Start (Fallback)"
        
        self.log(f"Phase 1 Post-process: 对齐时间轴 (Global 0 = {shift_mode})...")
        self.log(f"原点偏移量: {shift_target:.4f}s")
        
        # Shift all offsets
        for i in infos:
            old_off = i['res']['offset_seconds']
            new_off = old_off - shift_target
            i['res']['offset_seconds'] = new_off
            
            # Save duration to result to avoid re-probe in Phase 3
            i['res']['duration'] = i['res'].get('duration', self._get_fast_duration(i['res']['file']))
            
            # Update internal list
            
        with open(self.sync_info_path, 'w') as f:
            json.dump(alignment_results, f, indent=4)
            
        return alignment_results

    def _phase_2_smart_extraction(self, video_files, sync_results):
        """
        第二阶段：智能预提取 (Smart Extraction)
        根据同步结果计算每个视频需要提取的帧范围。
        """
        self.log("Phase 2: 智能预提取 (Smart Extraction)", header=True)
        
        # Simple Directory Check (Global) - Could be refined per video but let's keep it simple
        if self._check_smart_cache():
            return

        print("[Cache] 开始智能提取...")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        extractor = CacheExtractor(
            video_dir=self.video_dir,
            cache_dir=self.cache_dir,
            workers=self.workers,
            start_frame=None, # Will override per video
            end_frame=None,   # Will override per video
            buffer_seconds=self.buffer_seconds
        )
        
        # 1. Prepare Extraction Tasks
        extraction_tasks = []
        
        # Map sync results for easy lookup
        sync_map = {}
        if sync_results:
             for res in sync_results:
                 sync_map[os.path.basename(res['file'])] = res
                 
        # Determine Global Shift (needed to map output frames to global time)
        global_shift = 0.0
        if sync_results:
             # Find first chirp in reference video
             ref_fname = os.path.basename(video_files[0]) # Assume first is ref
             if ref_fname in sync_map:
                 res = sync_map[ref_fname]
                 ref_peaks = res.get('matched_points', {}).get('reference_peaks', [])
                 if ref_peaks:
                     first_chirp_time = ref_peaks[0]
                     # Current Global of this moment
                     current_global_chirp = first_chirp_time * res.get('drift_scale', 1.0) + res.get('offset_seconds', 0.0)
                     global_shift = current_global_chirp
        
        # Construct Video Info with Sync Params (Replicate logic from Finalize effectively)
        # We need this to map Output Frame -> Local Frame
        
        # We assume ref_fps is likely 30.0 or from first video
        # We need ref_fps to map start_frame (index) to time
        # Let's get ref_fps from first video
        syncer = AudioSyncSystem()
        ref_fps = 30.0
        if video_files:
            fps = syncer._get_video_fps(video_files[0])
            if fps: ref_fps = fps

        print(f"[Smart Extract] Target Output Range: {self.start_frame} - {self.end_frame} (Ref FPS: {ref_fps:.2f})")
        
        # Calculate Global Time Range for the expected output
        # Global Time = (FrameIdx - 1.0) / RefFPS? No, we defined MirrorTime = (Global * FPS) + 1.0
        # So Global = (MirrorTime - 1.0) / FPS
        # We want to extract frames corresponding to output indices start_frame to end_frame
        
        # Frame i -> MirrorTime i? No, Frame i corresponds to time T relative to start?
        # Let's align with Finalize logic:
        # Finalize: output frame idx `t_idx` (e.g. 0, 1, 2...)
        # target_time_relative = t_idx / ref_fps
        # target_global_time = intersect_start + target_time_relative 
        # Wait, intersect_start depends on valid intersection of all videos.
        # But we need to extract frames BEFORE we know the full intersection length? 
        # Actually Finalize calculates Intersection first.
        # If we do Extraction first, we might extract frames that are outside intersection?
        # Or we define output start/end relative to the "Aligned Global Origin" (Chirp).
        
        # User Args `start_frame` / `end_frame` usually mean "output frame 0 to N" of the FINAL sequence.
        # But the final sequence is defined by Intersection.
        # However, Intersection depends on Duration/Offset.
        # We DO have Offset now. We can calculate Intersection!
        
        # Let's calculate Intersection boundaries first to know what "Frame 0" corresponds to.
        # Replicate Intersection Logic (Simplified)
        
        vid_infos = {}
        for v_path in video_files:
            fname = os.path.basename(v_path)
            # Duration is needed for intersection end
            duration = 0.0
            if fname in sync_map and 'duration' in sync_map[fname]:
                 duration = sync_map[fname]['duration']
            else:
                 duration = self._get_fast_duration(v_path)
            
            offset = 0.0
            drift = 1.0
            if fname in sync_map:
                offset = sync_map[fname].get('offset_seconds', 0.0)
                drift = sync_map[fname].get('drift_scale', 1.0)
            
            # Apply Global Shift (Already normalized in Phase 1)
            
            global_start = offset
            global_end = (duration * drift) + offset
            
            vid_infos[fname] = {'offset': offset, 'drift': drift, 'fps': ref_fps, 'g_start': global_start, 'g_end': global_end}
            
        
        # Note: sync_results already normalized in Phase 1 so that Intersect Start = 0.0
        # But we recalculate here just to report and confirm
        all_starts = [v['g_start'] for v in vid_infos.values()]
        all_ends = [v['g_end'] for v in vid_infos.values()]
        
        intersect_start = max(all_starts) if all_starts else 0.0
        intersect_end = min(all_ends) if all_ends else 1.0
        
        print(f"[Smart Extract] Intersect Start (Normalized): {intersect_start:.4f}s, End: {intersect_end:.4f}s")
        print(f"[Smart Extract] User Range: Frame {self.start_frame} to {self.end_frame} (Relative to Intersect Start=0)")
        
        # Output Frame Index is now Absolute relative to Global 0 (which is now Intersect Start)
        # Frame 0 = 0.0s Global Time
        
        req_start_global = self.start_frame / ref_fps
        req_end_global = self.end_frame / ref_fps
        
        # Now map to Local Time for each video
        for v_path in video_files:
            fname = os.path.basename(v_path)
            info = vid_infos[fname]
            
            # Local = (Global - Offset) / Drift
            # We want to extract range [req_start, req_end]
            
            local_start_time = (req_start_global - info['offset']) / info['drift']
            local_end_time = (req_end_global - info['offset']) / info['drift']
            
            # Convert to Local Frame Index (Estimate)
            # We need actual FPS of this video to be precise, let's use syncer to get it or fallback
            v_fps = syncer._get_video_fps(v_path) or 30.0
            
            l_start_frame = int(local_start_time * v_fps)
            l_end_frame = int(local_end_time * v_fps)
            
            extraction_tasks.append({
                'path': v_path,
                's_frame': l_start_frame,
                'e_frame': l_end_frame
            })
            
        # 3. Execute Extraction
        # Estimate total for progress bar
        total_est = 0
        for t in extraction_tasks:
             # buffer approx
             buf_f = int(self.buffer_seconds * 30) * 2
             total_est += (t['e_frame'] - t['s_frame'] + 1) + buf_f
             
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        with tqdm(total=total_est, unit='img', desc="[Cache] Smart Extracting") as pbar:
             with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [
                    executor.submit(extractor._extract_video, t['path'], pbar, t['s_frame'], t['e_frame']) 
                    for t in extraction_tasks
                ]
                for _ in as_completed(futures):
                    pass


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



    def _phase_3_finalize(self, video_files, sync_results):
        """
        第三阶段：生成最终输出 (Intersection & Shift)
        """
        self.log("Phase 3: 同步抽帧", header=True)
        
        # 1. 构造 video info map
        video_info = {}
        syncer = AudioSyncSystem()
        
        sycned_map = {}
        if sync_results:
            for res in sync_results:
                fname = os.path.basename(res['file'])
                sycned_map[fname] = res
        
        self.log("计算视频原数据与验证公共区域...")
        
        from tqdm import tqdm
        # Optim: Duration cached, instant loop, no progress bar needed for Calc, but needed for Finalize
        for v_path in video_files:
            fname = os.path.basename(v_path)
            fps = syncer._get_video_fps(v_path)
            if fps is None: fps = 30.0
            
            # 优化：使用 Fast Probe 替代全量扫描
            # 如果 sync_results 里已经存了 duration (Phase 1)，直接用
            duration = 0.0
            if fname in sycned_map and 'duration' in sycned_map[fname]:
                 duration = sycned_map[fname]['duration']
                 # print(f"[Cache Hit] Duration for {fname}: {duration}")
            else:
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
            if fname in sycned_map:
                info['offset'] = sycned_map[fname].get('offset_seconds', 0.0)
                info['drift'] = sycned_map[fname].get('drift_scale', 1.0)
                info['is_synced'] = True
                
            info['global_start'] = info['offset']
            info['global_end'] = (info['duration'] * info['drift']) + info['offset']
            
            video_info[fname] = info

        
        # 2. 计算 Intersection
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
        print(f"[Smart Extract] Intersect Start (Normalized): {intersect_start:.4f}s, End: {intersect_end:.4f}s")
        print(f"[Smart Extract] User Range: Frame {self.start_frame} to {self.end_frame} (Relative to Intersect Start=0)")
        print("-" * 40 + "\n")
        
        if intersect_duration <= 0:
             self.log("警告：视频之间公共重叠区域极小或不存在！仍继续尝试输出...")

        # 3. 确定参考FPS
        ref_fps = 30.0
        if video_files:
            first_name = os.path.basename(video_files[0])
            ref_fps = video_info[first_name]['fps']
            
        # 4. 确定最终输出范围
        # User defined start_frame / end_frame is now ABSOLUTE relative to Global 0
        
        # Optional: Warn if user range is outside intersection
        user_start_s = self.start_frame / ref_fps
        user_end_s = self.end_frame / ref_fps
        
        if user_end_s < intersect_start or user_start_s > intersect_end:
             print("[警告] 请求的帧范围完全在公共重叠区域之外！生成的可能是黑屏或无效画面。")
        
        target_frames = list(range(self.start_frame, self.end_frame + 1))
        
        self.log(f"最终输出范围 (Frame Index): {self.start_frame} - {self.end_frame} (Ref FPS: {ref_fps:.2f})")
        self.log(f"对应全局时间: {user_start_s:.2f}s - {user_end_s:.2f}s")

        # 实例化 Fallback Extractor
        fallback_extractor = CacheExtractor(
            self.video_dir, self.cache_dir, workers=1
        )
        
        # Parallel Execution
        from concurrent.futures import ThreadPoolExecutor
        
        # Task: (target_frame_idx, video_path)
        all_tasks = []
        for t_idx in target_frames:
            for v_path in video_files:
                all_tasks.append((t_idx, v_path))

        # 定偏函数
        # 注意: intersect_start 不再用于偏移时间，仅传入 0.0 (因为 Frame 0 = Global 0.0)
        def process_func(task_args):
            t_idx, v_path = task_args
            self._process_single_image_task(
                target_frame_idx=t_idx,
                video_file=v_path,
                frame_dir_name=f"frame_{t_idx:06d}",
                intersect_start=0.0, # FIXED: Global 0.0 is Intersect Start
                video_info=video_info,
                fallback_extractor=fallback_extractor,
                ref_fps=ref_fps
            )

        self.log(f"开始生成帧 (多线程并发: {self.workers} workers)...")
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            list(tqdm(executor.map(process_func, all_tasks), total=len(all_tasks), desc="[Finalize] Processing"))



    def _process_single_image_task(self, target_frame_idx, video_file, frame_dir_name, intersect_start, video_info, fallback_extractor, ref_fps):
        """处理单个视频的单帧任务"""
        # Pick 3 deterministic videos for debugging
        debug_targets = sorted(list(video_info.keys()))[:3]
        
        fname = os.path.basename(video_file)
        v_data = video_info.get(fname)
        if not v_data: return

        # 目标时间 (相对于 Intersect Start/Global 0)
        # Target Frame 0 corresponds to Global 0.0s
        target_time_relative_to_output_start = target_frame_idx / ref_fps
        target_global_time = intersect_start + target_time_relative_to_output_start

        # 计算该视频的本地时间
        # Local = (Global - Offset) / Drift
        local_time = (target_global_time - v_data['offset']) / v_data['drift']
        
        # 计算本地帧索引
        local_frame_idx = int(local_time * v_data['fps'])

        # 构造缓存路径 (注意: CacheExtractor 使用扁平命名: video_frame_xxxx.jpg)
        vid_stem = os.path.splitext(fname)[0]
        cache_filename = f"{vid_stem}_frame_{local_frame_idx:06d}.jpg"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # 构造目标输出路径
        # 根据 output_structure 决定 dst_dir 和 dst_name
        if self.output_structure == 'by_frame':
            dst_dir = os.path.join(self.output_dir, frame_dir_name)
            dst_name = f"{vid_stem}.jpg"
        else:
            dst_dir = os.path.join(self.output_dir, vid_stem)
            dst_name = f"{frame_dir_name}.jpg"
        
        dst_path = os.path.join(dst_dir, dst_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 1. 获取目标文件 (Copy or Fallback)
        # 注意：CacheExtractor (和 Fallback) 会将原始 PTS 写入 "MirrorTime"
        # 1. 获取目标文件 (Copy or Fallback)
        # 简化逻辑：Cache Hit -> Copy, Cache Miss -> Extract
        if os.path.exists(cache_path):
            try:
                shutil.copy2(cache_path, dst_path)
            except Exception as e:
                if fname in debug_targets:
                    print(f"[Cache Copy Error] {fname}: {e}")
        else:
            # Fallback Extract
            # 提取到目标位置 (写入默认 Raw PTS)
            fallback_extractor.extract_single_frame_fallback(
                video_file, local_frame_idx, dst_path, {} 
            )
        
        # 无论 Hit 还是 Fallback，现在的 dst_path 文件里应该都有 Raw PTS (MirrorTime)
        # 我们统一读取并复写
        
        raw_pts = 0.0
        read_success = False
        
        # Method A: PIL getexif (Fast)
        try:
            with Image.open(dst_path) as img:
                exif = img.getexif()
                if 0x9286 in exif:
                    comment_data = exif[0x9286]
                    if isinstance(comment_data, bytes):
                        # Handle ASCII header
                        if comment_data.startswith(b'ASCII\0\0\0'):
                            comment_data = comment_data[8:]
                        comment_str = comment_data.decode('utf-8', errors='ignore').strip('\x00')
                    else:
                        comment_str = str(comment_data)
                        
                    # Parse JSON
                    comment_str = comment_str.strip()
                    if comment_str:
                        meta = json.loads(comment_str)
                        raw_pts = float(meta.get("MirrorTime", 0.0))
                        read_success = True
        except Exception as e:
            # PIL read failed, will try piexif
            pass

        # Method B: piexif (Robust)
        if not read_success:
            try:
                import piexif
                # piexif.load returns a dict with "Exif", "0th", etc.
                exif_dict = piexif.load(dst_path)
                if "Exif" in exif_dict and piexif.ExifIFD.UserComment in exif_dict["Exif"]:
                    comment_data = exif_dict["Exif"][piexif.ExifIFD.UserComment]
                    if isinstance(comment_data, bytes):
                        if comment_data.startswith(b'ASCII\0\0\0'):
                            comment_data = comment_data[8:]
                        comment_str = comment_data.decode('utf-8', errors='ignore').strip('\x00')
                        comment_str = comment_str.strip()
                        if comment_str:
                            meta = json.loads(comment_str)
                            raw_pts = float(meta.get("MirrorTime", 0.0))
                            read_success = True
            except Exception as e:
                 if fname in debug_targets:
                     print(f"[Meta Read Error {fname}] Methods failed. Last error: {e}")

        
        # 如果读取失败（理论不应发生），回退到估算
        if not read_success:
            if fname in debug_targets:
                 print(f"[Meta Warn {fname}] Read FAILED from {os.path.basename(cache_path)}. Using Fallback Calculation.")
            
            # 使用计算得到的 local_time，而不是 local_frame_idx / fps
            # 因为 int() 舍入会导致精度损失，local_time 更准确
            raw_pts = local_time
        else:
             if fname in debug_targets:
                 # Debug: Confirm what we read
                 pass
            
        # 计算 Sync Time
        # Global = Raw * Drift + Offset
        if v_data['is_synced']:
            abs_global = raw_pts * v_data['drift'] + v_data['offset']
        else:
            abs_global = raw_pts
            
        # MirrorTime is 1-based: T=0 -> MirrorTime=1.0
        final_time = (abs_global * ref_fps) + 1.0
        
        # 覆写
        self._inject_metadata(dst_path, {"MirrorTime": final_time})
        
        # DEBUG: Print MirrorTime logic for select videos
        # Pick 3 deterministic videos for debugging
        if fname in debug_targets:
             calc_details = f"({raw_pts:.10f} * {v_data['drift']:.10f}) + {v_data['offset']:.10f} -> Global {abs_global:.10f}"
             final_step = f"({abs_global:.10f} * {ref_fps:.5f}) + 1.0 -> {final_time:.10f}"
             print(f"\n[Debug {fname}]\n  Calc: {calc_details}\n  Final: {final_step}")

    def _inject_metadata(self, image_path, meta_dict):
        """写入 EXIF UserComment (使用 piexif 以避免重编码)"""
        try:
            import piexif
            
            meta_json = json.dumps(meta_dict)
            
            # 手动构建 UserComment bytes (ASCII header)
            # 这种方式最稳健，兼容性最好
            user_comment = b"ASCII\0\0\0" + meta_json.encode("utf-8")
            
            # 加载现有 EXIF 或创建新的
            exif_dict = {"Exif": {piexif.ExifIFD.UserComment: user_comment}}
            try:
                # 尝试保留原有 EXIF (如果有)
                old_exif = piexif.load(image_path)
                if "Exif" in old_exif:
                    old_exif["Exif"][piexif.ExifIFD.UserComment] = user_comment
                    exif_dict = old_exif
                else:
                    old_exif["Exif"] = {piexif.ExifIFD.UserComment: user_comment}
                    exif_dict = old_exif
            except Exception:
                # 加载失败则使用新的
                pass
            
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            
        except ImportError:
            print("[Error] 请先安装 piexif: pip install piexif")
        except Exception as e:
            print(f"[Meta Inject Error] {image_path}: {e}")

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

        # 1. 同步 (Phase 1)
        if self.skip_sync:
            self.log("Phase 1: 用户请求跳过音频同步 (Professional Sync Mode)")
            sync_results = None # Triggers default fallback in subsequent phases
        else:
            sync_results = self._phase_1_audio_sync(video_files)
        
        # 2. 预提取 (Phase 2 - Smart)
        self._phase_2_smart_extraction(video_files, sync_results)
        
        # 3. 输出 (Phase 3: 同步抽帧)
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
    
    parser.add_argument("--start_frame", type=int, default=200, help="目标起始帧")
    parser.add_argument("--end_frame", type=int, default=200, help="目标结束帧")
    parser.add_argument("--structure", choices=['by_frame', 'by_video'], default='by_frame', help="输出结构")
    parser.add_argument("--workers", type=int, default=6, help="线程数")
    parser.add_argument("--buffer", type=float, default=0, help="预提取缓冲时长(秒)")
    # 保留部分 Sync 参数
    parser.add_argument("--window", type=float, default=3.0, help="Sync Window")
    parser.add_argument("--skip_sync", action="store_true", help="跳过音频同步 (假设已经是硬件同步的)")
    
    args = parser.parse_args()

    pipeline = FullSyncPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        output_structure=args.structure,
        workers=args.workers,
        buffer_seconds=args.buffer,
        matching_window=args.window,
        skip_sync=args.skip_sync
    )
    
    pipeline.run()
