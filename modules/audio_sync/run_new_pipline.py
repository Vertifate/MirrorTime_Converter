
import os
import argparse
import json
import enum
import sys
import shutil
import numpy as np
import ffmpeg
import subprocess
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# 尝试导入音频同步模块
try:
    from audio_sync import AudioSyncSystem
except ImportError:
    # 如果在当前目录找不到，尝试添加当前目录并重新导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from audio_sync import AudioSyncSystem

# 尝试导入 piexif
try:
    import piexif
except ImportError:
    print("警告: 未找到 piexif 模块，MirrorTime 元数据可能无法写入图像。")
    piexif = None

class GlobalTimelineConfig:
    def __init__(self, start_time: float = None, end_time: float = None, framerate: float = 30.0):
        self.start_time = start_time
        self.end_time = end_time
        self.framerate = framerate

    @property
    def total_frames(self):
        # 如果未初始化时间，返回0
        if self.start_time is None or self.end_time is None:
            return 0
        duration = self.end_time - self.start_time
        if duration <= 0: return 0
        return int(duration * self.framerate) + 1

    def get_global_timestamps(self):
        return np.linspace(
            self.start_time, 
            self.end_time, 
            self.total_frames, 
            endpoint=True
        )

class VideoSyncInfo:
    def __init__(self, file_path, drift_scale=1.0, offset_seconds=0.0):
        self.file_path = file_path
        self.drift_scale = drift_scale  # 修正系数
        self.offset_seconds = offset_seconds # 本地偏移
        self.basename = os.path.basename(file_path)

    def global_to_local(self, global_time):
        """
        Global = (Local * Drift) + Offset
        => Local * Drift = Global - Offset
        => Local = (Global - Offset) / Drift
        """
        if self.drift_scale == 0: return 0
        return (global_time - self.offset_seconds) / self.drift_scale

    def local_to_global(self, local_time):
        return local_time * self.drift_scale + self.offset_seconds

class PipelineManager:
    def __init__(self, video_dir, output_dir, use_sync, 
                 global_timeline_cfg: GlobalTimelineConfig, 
                 max_workers=None, use_cuda=False):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.use_sync = use_sync
        self.global_cfg = global_timeline_cfg
        self.max_workers = max_workers or os.cpu_count() or 4
        self.use_cuda = use_cuda
        
        self.video_files = self._scan_videos()
        self.video_sync_infos = []
        
        # Audio Params Default
        self.audio_params = {
            'chirp_duration': 0.5,
            'start_freq': 1000,
            'end_freq': 8000,
            'sample_rate': 48000,
            'window': 3.0
        }

    def set_audio_params(self, chirp_duration, start_freq, end_freq, sample_rate, window):
        self.audio_params = {
            'chirp_duration': chirp_duration,
            'start_freq': start_freq,
            'end_freq': end_freq,
            'sample_rate': sample_rate,
            'window': window
        }

    def _scan_videos(self):
        video_exts = ['.mov', '.mp4', '.avi', '.m4v', '.mkv']
        files = []
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"视频目录不存在: {self.video_dir}")
        for f in sorted(os.listdir(self.video_dir)):
            if os.path.splitext(f)[1].lower() in video_exts:
                files.append(os.path.join(self.video_dir, f))
        return files

    def run_synchronization(self):
        """步骤1: 运行同步计算或应用默认值"""
        print("\n" + "="*30)
        print("步骤 1/3: 全局时间轴计算")
        print("="*30)

        if not self.use_sync:
            print("提示: 用户选择跳过音频同步。")
            print("      所有视频将假定为已对齐 (Offset=0, Drift=1.0)。")
            for v in self.video_files:
                self.video_sync_infos.append(VideoSyncInfo(v, 1.0, 0.0))
            
            # Non-Sync 模式下的默认值处理
            if self.global_cfg.start_time is None: self.global_cfg.start_time = 0.0
            if self.global_cfg.end_time is None: 
                print("警报: 在非同步模式下未指定 global_end，默认设为 10.0s")
                self.global_cfg.end_time = 10.0
            return

        # 实例化音频同步系统
        print(f"正在分析 {len(self.video_files)} 个视频的音频同步信号...")
        print(f"参数: {self.audio_params}")
        
        syncer = AudioSyncSystem(
            chirp_duration=self.audio_params['chirp_duration'],
            start_freq=self.audio_params['start_freq'],
            end_freq=self.audio_params['end_freq'],
            sample_rate=self.audio_params['sample_rate']
        )
        
        # 运行对齐
        # audio_sync.py 的 align_videos 返回一个包含 offset/drift 的列表
        results = syncer.align_videos(
            self.video_files, 
            matching_window_seconds=self.audio_params['window'],
            visualize=False, 
            tqdm_desc="分析音频信号"
        )

        if not results:
            print("\n[错误] 未检测到声音同步信号，请检查视频数据是否需要同步。")
            print("       程序将尝试使用默认对齐继续 (Offset=0, Drift=1.0)，")
            print("       但结果可能不正确。")
            # Fallback
            for v in self.video_files:
                self.video_sync_infos.append(VideoSyncInfo(v, 1.0, 0.0))
            return

        # --- 全局时间轴校正 ---
        # 目标: 第一次声音快板为0 (Global Time 0)
        # audio_sync.py 默认以第一个视频为参考，offset=0
        # 它的 offset 定义通常是 target_start - ref_start
        # 我们需要找到所有视频中发生的"对齐后的绝对时间"的"第一个共同事件" 并将其设为 0
        
        # AudioSyncSystem 结果包含 matched_points['reference_peaks']
        # 我们可以找到参考视频的第一个 peak，比如是 T_ref_peak_0
        # 如果我们希望这就是全局 0 点，那么全局偏移 Global_Shift = T_ref_peak_0
        
        # 获取参考视频的第一个峰值作为全局零点基准
        ref_result = results[0]
        ref_peaks = ref_result.get('matched_points', {}).get('reference_peaks', [])
        
        global_zero_shift = 0.0
        if ref_peaks:
            global_zero_shift = ref_peaks[0]
            last_peak = ref_peaks[-1]
            max_global_limit = last_peak - global_zero_shift
            
            print(f"检测到全局参考零点 (第一次快板): {global_zero_shift:.4f}s (参考视频时间)")
            print(f"检测到全局终点 (最后一次快板): {max_global_limit:.4f}s (相对时长)")
            
            # --- 动态范围处理 ---
            # 如果 global_start 为 None (默认), 设为 0.0
            if self.global_cfg.start_time is None:
                 print(f"[提示] 未指定 global_start，默认设为 0.0s")
                 self.global_cfg.start_time = 0.0

            # 如果 global_end 为 None (默认), 设为 max_global_limit
            if self.global_cfg.end_time is None:
                 print(f"[提示] 未指定 global_end，默认设为最大全局时间轴: {max_global_limit:.4f}s")
                 self.global_cfg.end_time = max_global_limit
            
            # 强制限制抽帧范围 (仅当已指定且超出时)
            if self.global_cfg.end_time > max_global_limit:
                print(f"[提示] 设定的结束时间 ({self.global_cfg.end_time:.4f}s) 超出了最大全局时间轴。")
                print(f"       -> 已强制限制抽帧范围至 {max_global_limit:.4f}s。")
                self.global_cfg.end_time = max_global_limit
        else:
             print("警告: 即使对其成功也未找到参考峰值，无法执行全局零点校正。")

        # 填充 VideoSyncInfo
        # AudioSync 算出的 offset 是 "Video - Reference" 还是 "Reference - Video"?
        # audio_sync.py: offset = ransac.intercept_  (Target = Ref * Drift + Offset) -> Target is Y? No.
        # audio_sync: ransac.fit(X, y). X=Current(Target), y=Reference.
        # So: Reference = Current * Drift + Offset
        # So: Global(Ref-based) = Local * Drift + Offset
        
        # 更新 Offset，使得 Global = 0 at first peak.
        # Current_Global = Local * Drift + Old_Offset
        # New_Global = Current_Global - Global_Zero_Shift
        #            = Local * Drift + (Old_Offset - Global_Zero_Shift)
        # So New_Offset = Old_Offset - Global_Zero_Shift
        
        for res in results:
            path = res['file']
            old_drift = res.get('drift_scale', 1.0)
            old_offset = res.get('offset_seconds', 0.0)
            
            new_offset = old_offset - global_zero_shift
            
            self.video_sync_infos.append(VideoSyncInfo(path, old_drift, new_offset))
            
            # print(f"  > {os.path.basename(path)}: Drift={old_drift:.6f}, GlobalOffset={new_offset:.4f}s")

    def run_frame_extraction(self):
        """步骤2 & 3: 从全局时间轴逆算并提取"""
        print("\n" + "="*30)
        print("步骤 2/3: 准备抽帧计划")
        print("="*30)

        # 全局时间点
        global_timestamps = self.global_cfg.get_global_timestamps()
        total_global_frames = len(global_timestamps)
        print(f"全局时间轴配置: {self.global_cfg.start_time}s -> {self.global_cfg.end_time}s, FPS={self.global_cfg.framerate}")
        print(f"总计需要提取的全局帧数: {total_global_frames}")

        if total_global_frames == 0:
            print("警告: 需要提取的帧数为0。")
            return

        # 整理抽帧任务: 按视频分组，避免对同一个视频反复开关 ffmpeg
        # 我们采用 "批量区间提取 + 精确选择" 的策略来平衡效率
        # 但为了极高的精确度和避免预扫描所有 PTS，我们使用 filter 'select' 配合 'showinfo'
        
        # 构建任务列表
        # task = { video_path: [ {target_local_time, global_frame_index, global_timestamp}, ... ] }
        extraction_tasks = { v.file_path: [] for v in self.video_sync_infos }
        
        for g_idx, g_time in enumerate(global_timestamps):
            for v_info in self.video_sync_infos:
                local_time = v_info.global_to_local(g_time)
                # 只有 local_time >= 0 才通常有效，但有些视频可能有负的逻辑时间(未开始录制)
                # 我们假设如果 < 0 则该视频在该全局时刻无图像，或提取第一帧
                # 这里只提取 local_time >= 0 的
                if local_time < 0:
                    continue # Video hasn't started yet relative to global zero
                
                # 我们不设定上限，假设视频够长。如果超出 ffmpeg 会提取失败或空，我们在 worker 里处理
                extraction_tasks[v_info.file_path].append({
                    'target_local': local_time,
                    'global_idx': g_idx,
                    'global_time': g_time
                })

        # 移除空任务
        tasks_list = [ (path, items) for path, items in extraction_tasks.items() if items ]
        
        print(f"\n" + "="*30)
        print("步骤 3/3: 执行并行抽帧与元数据注入")
        print("="*30)
        
        # 进度条基于 "处理的全局帧 X 视频数"，即生成的图片总数
        total_images = sum(len(items) for _, items in tasks_list)
        
        with tqdm(total=total_images, unit="img", desc="Processing") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交每个视频的处理任务
                futures = []
                for v_path_str, items in tasks_list:
                    # 获取对应的 info 对象
                    v_info = next(info for info in self.video_sync_infos if info.file_path == v_path_str)
                    futures.append(executor.submit(self._process_single_video, v_info, items, pbar))
                
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"\n[任务异常] {e}")

    def _process_single_video(self, v_info: VideoSyncInfo, items, pbar):
        """
        处理单个视频的所有抽帧请求
        items: list of {'target_local', 'global_idx', 'global_time'}
        """
        if not items: return

        # 按时间排序
        items.sort(key=lambda x: x['target_local'])
        
        # 分块处理以防止命令行参数过长
        # 且使用 select='between(...)' 可能更好，或者如果有明确的时间点，使用 select='eq(n,xxx)' 需要知道索引
        # 由于我们不知道帧索引，只能用 select='gte(t,T)*lte(t,T+delta)' 
        # 或者使用 select='between(t, A, B)' 拉取整个段，然后选最近的
        
        # 优化策略:
        # 既然是连续抽帧 (FPS通常固定)，我们可以计算大概的区间
        # 比如 items 覆盖了 1.0s -> 5.0s, 我们就提取 ffmpeg -ss 0.5 -to 5.5 ... select='between(t, 1.0, 5.0)'
        # 并开启 showinfo 获取每一帧的真实 PTS
        # 然后在内存中做 "Ideal -> Real" 的最近匹配
        
        # 获取该视频所有请求的时间范围
        min_t = items[0]['target_local']
        max_t = items[-1]['target_local']
        
        # 加上一点缓冲 (Buffer)
        buffer = 0.5
        start_scan = max(0, min_t - buffer)
        end_scan = max_t + buffer
        
        # 构建输出目录 cache
        vid_basename = os.path.splitext(v_info.basename)[0]
        temp_dir = os.path.join(self.output_dir, ".temp_extract", vid_basename)
        os.makedirs(temp_dir, exist_ok=True)
        
        # FFmpeg 命令
        # select='between(t, start, end)'
        # 配合 showinfo
        
        # 临时文件 Pattern
        # 注意: jpg q:v 表示质量
        temp_pattern = os.path.join(temp_dir, "%06d.jpg")
        
        try:
            # 使用 ffmpeg-python 构建
            # 注意: 如果视频非常长，我们要用 -ss 进行 seek 以加速 (input option)
            # 使用 -ss 在 input 之前是 Keyframe seek，非常快
            
            # select 表达式中的 t 是相对于 input start 的，还是 absolute 的?
            # 如果用了 -ss，t 通常重置为 0，除非用 -copyts。
            # 为简单起见，且为了精度，我们通常不用 seek 或者小心计算 offset。
            # 为了准确性，我们不使用 -ss (除非我们确信), 或者我们只在 start_scan 很大时使用
            # 考虑到效率，必须使用 seek
            
            seek_time = max(0, start_scan - 0.5) # 提前0.5秒 seek 以确保关键帧并留缓冲 (用户修改: 5.0 -> 0.5)
            
            # 修正 select 时间: 如果 seek 了，t 会从 0 开始算 (如果不加 -copyts)
            # 为了逻辑简单，我们不使用 -copyts，而是调整 filter 时间
            # relative_start = start_scan - seek_time
            # relative_end = end_scan - seek_time
            # filter: between(t, relative_start, relative_end)
            
            # 尝试开启 GPU 加速
            # 这里简单硬编码，如果环境中支持 cuda 则会加速，否则可能会报错或回退（取决于 FFmpeg 编译）
            # 为了安全，我们通过参数传入或者先检测。这里默认尝试添加 -hwaccel cuda
            # 如果失败率高，建议由用户参数控制。
            
            ffmpeg_input_args = {'ss': seek_time}
            if self.use_cuda:
                ffmpeg_input_args['hwaccel'] = 'cuda'

            cmd = (
                ffmpeg
                .input(v_info.file_path, **ffmpeg_input_args)
                .filter('select', f'between(t,{start_scan - seek_time},{end_scan - seek_time})')
                .filter('showinfo')
                .output(temp_pattern, vsync=0, **{'q:v': 2})
                .overwrite_output()
            )
            
            # 运行并捕获 stderr 以解析 showinfo
            try:
                out, err = cmd.run(capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                print(f"[FFmpeg Error] {v_info.basename}: {e.stderr.decode('utf8', errors='ignore')}")
                pbar.update(len(items)) 
                return

            # 解析 stderr 中的 PTS
            # 格式: [Parsed_showinfo_1 @ ...] n:  15 pts:  12800 pts_time:0.833333 pos:  ...
            # 注意这些 time 是相对于 seek_time 的 (如果没 copyts)
            
            extracted_frames = [] # (index, relative_time, absolute_time)
            
            err_str = err.decode('utf-8', errors='ignore')
            
            timestamps = []
            # 正则匹配
            for line in err_str.splitlines():
                if "pts_time:" in line:
                    m = re.search(r"pts_time:\s*([\d\.]+)", line)
                    if m:
                        rel_t = float(m.group(1))
                        abs_t = rel_t + seek_time
                        timestamps.append(abs_t)
            
            # 检查提取的文件
            # temp files are 000001.jpg, 000002.jpg ... (ffmpeg default sequence start 1)
            # showinfo 顺序应该和文件顺序一致
            
            files = sorted(os.listdir(temp_dir))
            
            # 匹配
            # files[i] 对应 timestamps[i]
            
            if not files:
                # print(f"警告: {v_info.basename} 未提取到任何帧 (范围 {start_scan}-{end_scan})")
                pbar.update(len(items))
                shutil.rmtree(temp_dir, ignore_errors=True)
                return

            if len(files) != len(timestamps):
                # 这是一个经典问题，有时 ffmpeg drop frame 但 showinfo 还在? 
                # 通常 vsync=0 会保持一致。如果不对齐，我们只能尽力而为。
                # 简单处理：取最小长度
                limit = min(len(files), len(timestamps))
                files = files[:limit]
                timestamps = timestamps[:limit]

            # 将 timestamps 转为 numpy 方便查询
            ts_arr = np.array(timestamps)
            
            # 对每个 ideal item 找最近的 real frame
            for item in items:
                ideal_t = item['target_local']
                
                # 找最近
                idx = (np.abs(ts_arr - ideal_t)).argmin()
                real_t = ts_arr[idx]
                min_dist = abs(real_t - ideal_t)
                
                # 阈值检查：如果误差太大(例如 > 0.5秒)，说明可能没提取到想要的帧
                if min_dist > 0.5:
                    # 获取失败
                    pbar.update(1)
                    continue
                
                # 对应的文件
                source_file = os.path.join(temp_dir, files[idx])
                
                # 目标路径: output_dir / global_frame_{g_t:.3f} / basename.jpg (或者自定义命名)
                # 用户要求: "文件夹的命名上，framexxxxxx也应该为全局对应帧"
                # 全局对应帧号 (整数) = round(global_time * fps)
                g_time = item['global_time']
                fps = self.global_cfg.framerate
                global_frame_int = int(round(g_time * fps))
                
                # 文件夹命名: frame_{global_integer}
                frame_dir_name = f"frame_{global_frame_int:06d}"
                frame_out_dir = os.path.join(self.output_dir, frame_dir_name)
                os.makedirs(frame_out_dir, exist_ok=True)
                
                out_name = os.path.splitext(v_info.basename)[0] + ".jpg"
                out_path = os.path.join(frame_out_dir, out_name)
                
                # 优化: 合并 "复制文件" 与 "注入元数据" 为一步
                # 直接读取源文件 Image，注入元数据后保存到目标路径，避免两次 IO 写操作
                self._save_with_metadata(source_file, out_path, v_info, ideal_t, real_t, g_time, fps)
                
                pbar.update(1)

        except Exception as e:
            print(f"[处理异常] {v_info.basename}: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(len(items))
        finally:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _save_with_metadata(self, src_path, dst_path, v_info, ideal_local, real_local, ideal_global_ts, fps):
        """
        读取 src_path，注入 MirrorTime 元数据，直接保存到 dst_path。
        减少了一次 shutil.copy 的 IO 开销。
        """
        # Calculate Real Global Time
        real_global_ts = v_info.local_to_global(real_local)
        
        # MirrorTime: 全局精确帧号 (带小数) = Real Global Time * FPS
        mirror_time_frame = real_global_ts * fps
        
        meta = {
            "MirrorTime": mirror_time_frame,
            "GlobalTimestamp": real_global_ts,
            "LocalTimestamp": real_local,
            "IdealGlobalFrame": int(round(ideal_global_ts * fps)),
            "VideoSource": v_info.basename
        }
        
        try:
            # 准备 EXIF
            exif_bytes = None
            if piexif:
                try:
                    meta_json = json.dumps(meta)
                    user_comment = b"ASCII\0\0\0" + meta_json.encode("utf-8")
                    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: user_comment}}
                    exif_bytes = piexif.dump(exif_dict)
                except Exception as e:
                    pass

            with Image.open(src_path) as img:
                # 保存到目标路径
                # quality=95 保证高质量，subsampling=0 关闭色度抽样以保真
                if exif_bytes:
                    img.save(dst_path, exif=exif_bytes, quality=95, subsampling=0)
                else:
                    img.save(dst_path, quality=95, subsampling=0)
                    
        except Exception as e:
            # 如果一般保存失败，降级为直接复制
            # print(f"Error saving image {dst_path}: {e}, falling back to copy.")
            shutil.copy2(src_path, dst_path)

def main():
    parser = argparse.ArgumentParser(description="MirrorTime 全流程处理管线", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 路径参数
    parser.add_argument("--video_dir", required=True, help="输入视频文件夹路径")
    parser.add_argument("--output_dir", required=True, help="输出结果保存路径")
    
    # 同步控制 (Sync Control)
    parser.add_argument("--use_sync", action='store_true', help="是否启用音频同步 (默认: False)")
    
    # 声音同步参数 (Audio Sync Parameters)
    sync_group = parser.add_argument_group('Audio Sync Configuration', '仅在启用 --use_sync 时生效')
    sync_group.add_argument("--chirp_duration", type=float, default=0.3, help="同步信号(Chirp)持续时间")
    sync_group.add_argument("--start_freq", type=int, default=1000, help="Chirp起始频率 (Hz)")
    sync_group.add_argument("--end_freq", type=int, default=8000, help="Chirp终止频率 (Hz)")
    sync_group.add_argument("--sample_rate", type=int, default=48000, help="音频采样率")
    sync_group.add_argument("--window", type=float, default=3.0, help="同步匹配窗口大小(秒) - 在此范围内寻找匹配的快板")
    
    # 全局时间轴参数 (Global Timeline Parameters)
    # 解释: 
    # 1. 0时刻 = 第一次检测到的快板/Chirp信号。
    # 2. 'Max Global Time' (Last Clapper) = 最后一次检测到的快板。
    # 3. 关于抽帧范围 (global_start/end):
    #    - 若启用同步 (--use_sync): 程序会强制将 'global_end' 限制在 'Last Clapper' 时刻内，以保证精度。
    #    - 若不启用同步: 您可以自由设置范围，甚至外推。
    #    - 'global_start' 可以为负数 (例如提取快板前的画面)。
    timeline_group = parser.add_argument_group('Global Timeline Extraction', '定义希望提取的全局时间范围 (0.0 = 第一次快板)')
    timeline_group.add_argument("--global_start", type=float, default=0, help="抽帧起始时间 (默认为0.0)")
    timeline_group.add_argument("--global_end", type=float, default=0.1, help="抽帧结束时间 (默认为最大全局时间)")
    timeline_group.add_argument("--fps", type=float, default=30.0, help="全局抽帧频率 (FPS)")
    
    # 性能参数
    parser.add_argument("--workers", type=int, default=8, help="并行处理线程数")
    parser.add_argument("--cuda", action='store_true', help="尝试使用 NVENC/CUDA 硬件加速解码")
    
    args = parser.parse_args()

    # 1. 配置检查
    if not os.path.isdir(args.video_dir):
        print(f"Error: 视频目录 '{args.video_dir}' 不存在。")
        return

    config = GlobalTimelineConfig(
        start_time=args.global_start,
        end_time=args.global_end,
        framerate=args.fps
    )

    manager = PipelineManager(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        use_sync=args.use_sync,
        global_timeline_cfg=config,
        max_workers=args.workers,
        use_cuda=args.cuda
    )
    
    # 注入额外的音频参数给 manager (需要修改 PipelineManager 接收这些参数，或者直接传给 AudioSyncSystem)
    if args.use_sync:
        manager.set_audio_params(
            chirp_duration=args.chirp_duration,
            start_freq=args.start_freq,
            end_freq=args.end_freq,
            sample_rate=args.sample_rate,
            window=args.window
        )

    # 2. 运行
    try:
        manager.run_synchronization()
        manager.run_frame_extraction()
        print("\n[完成] 所有任务已结束。")
    except KeyboardInterrupt:
        print("\n[中断] 用户强制停止。")
    except Exception as e:
        print(f"\n[失败] 程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
