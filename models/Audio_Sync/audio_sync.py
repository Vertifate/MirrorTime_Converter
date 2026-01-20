import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.linear_model import RANSACRegressor
import os
import argparse
import json
import ffmpeg
import sys
from tqdm import tqdm
from sync_detector import SingleVideoSyncDetector

class AudioSyncSystem:
    def __init__(self, chirp_duration=0.5, start_freq=1000, end_freq=8000, sample_rate=48000):
        """
        初始化同步系统
        :param chirp_duration: 扫频信号持续时间(秒)
        :param start_freq: 起始频率 (Hz) - 避开低频人声噪音(通常<500Hz)
        :param end_freq: 终止频率 (Hz) - 根据麦克风响应，通常8kHz足够
        :param sample_rate: 采样率
        """
        # 初始化底层的单视频检测器
        self.detector = SingleVideoSyncDetector(chirp_duration, start_freq, end_freq, sample_rate)
        self.sr = sample_rate # 保留 sr 属性供 save_reference_audio 使用

    def save_reference_audio(self, filename="sync_signal.wav", interval_silence=5.0):
        """
        生成用于现场播放的音频文件（包含多次重复，中间有间隔）
        :param interval_silence: 两次信号之间的静默间隔(秒)
        """
        if interval_silence > 0:
            silence = np.zeros(int(self.sr * interval_silence))
            # 组合：Chirp + 静音
            pattern = np.concatenate((self.detector.reference_chirp, silence))
        else:
            # 只包含Chirp信号
            pattern = self.detector.reference_chirp
        # 放大到 16-bit 范围
        audio_data = np.int16(pattern * 32767)
        wavfile.write(filename, self.sr, audio_data)
        print(f"[生成] 同步信号已保存为 {filename}，请在现场循环播放。")

    def _get_video_fps(self, video_path):
        """使用 ffmpeg-python 获取视频的平均帧率。"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream and 'avg_frame_rate' in video_stream:
                rate_str = video_stream['avg_frame_rate']
                if '/' in rate_str:
                    num, den = map(int, rate_str.split('/'))
                    return num / den if den != 0 else None
                else:
                    return float(rate_str)
            print(f"[警告] 在 {os.path.basename(video_path)} 中未找到视频流或FPS信息。")
            return None
        except Exception as e:
            print(f"无法获取 {os.path.basename(video_path)} 的FPS: {e}")
            return None

    def align_videos(self, video_paths, matching_window_seconds=1.0, reference_path=None, visualize=False, tqdm_desc=None):
        """
        计算多个视频相对于第一个视频的对齐参数
        :param video_paths: 视频文件路径列表
        :param visualize: 是否显示可视化图表
        """
        if not video_paths:
            return
        
        # 如果指定了参考视频，则将其移动到列表的第一个位置
        if reference_path and reference_path in video_paths:
            video_paths.insert(0, video_paths.pop(video_paths.index(reference_path)))
        
        # 如果需要可视化，检查 matplotlib 是否安装
        if visualize:
            try:
                import matplotlib
                matplotlib.use('Agg') # <-- Set backend BEFORE importing pyplot
                import matplotlib.pyplot as plt
            except ImportError:
                print("警告: 需要安装 matplotlib 才能使用可视化功能。请运行: pip install matplotlib")
                visualize = False # 禁用可视化


        if not tqdm_desc:
            print("-" * 50)
        reference_peaks = None
        results = []

        # 使用 tqdm 包装循环
        video_iterator = tqdm(enumerate(video_paths), total=len(video_paths), desc=tqdm_desc) if tqdm_desc else enumerate(video_paths)


        for i, v_path in video_iterator:
            if not os.path.exists(v_path):
                print(f"[错误] 文件不存在: {v_path}")
                if i == 0:
                    print("[终止] 参考视频丢失，无法继续对齐。")
                    return []
                continue

            if not tqdm_desc:
                print(f"[处理] 正在读取视频: {v_path} ...")
            audio_data = self.detector.extract_audio_from_video(v_path)
            if audio_data is None:
                if i == 0:
                    print(f"[终止] 无法从参考视频 {v_path} 提取音频。")
                    return []
                continue

            # 检测峰值
            peaks, corr_signal = self.detector.find_peaks(audio_data)
            if not tqdm_desc:
                print(f"视频 {os.path.basename(v_path)} 检测到 {len(peaks)} 个同步信号。")
            
            # 如果开启了可视化
            if visualize:
                plt.figure(figsize=(15, 5))
                time_axis = np.linspace(0, len(corr_signal) / self.sr, num=len(corr_signal))
                plt.plot(time_axis, corr_signal, label='Correlation Signal')
                # 在峰值位置绘制红色'x'标记
                peak_indices = (peaks * self.sr).astype(int)
                plt.plot(peaks, corr_signal[peak_indices], "x", color='r', markersize=10, label='Detected Peaks')
                plt.title(f"Correlation and Detected Peaks for: {os.path.basename(v_path)}")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Normalized Correlation")
                plt.legend()
                plt.grid(True)
                output_filename = f"peaks_{os.path.basename(v_path)}.png"
                plt.savefig(output_filename)
                plt.close() 
                if not tqdm_desc:
                    print(f"已将峰值检测图表保存为 {output_filename}")

            if i == 0:
                reference_peaks = peaks
                
                results.append({
                    "file": v_path,
                    "offset_seconds": 0.0,
                    "drift_scale": 1.0,
                    "matched_points": {
                        "reference_peaks": peaks.tolist(),
                        "target_peaks": peaks.tolist()
                    }
                })
                if not tqdm_desc:
                    print(f"设定 {os.path.basename(v_path)} 为主参考时间轴。")
            else:
                if len(peaks) < 1 or len(reference_peaks) < 1:
                    print(f"警告: 无法在 {v_path} 中找到足够的同步信号。")
                    continue

                # --- 鲁棒性对齐 (RANSAC) ---
                # 新匹配逻辑：对参考视频的每个峰值，在当前视频的 +/- N 秒窗口内寻找最佳匹配
                matched_X = []  # 当前视频中匹配上的峰值时间
                matched_y = []  # 参考视频中匹配上的峰值时间
                
                available_peaks = list(peaks)  # 创建一个可修改的峰值列表副本

                for ref_peak in reference_peaks:
                    best_match = None
                    min_diff = float('inf')

                    # 在指定窗口内寻找最近的峰值
                    for peak in available_peaks:
                        diff = abs(peak - ref_peak)
                        if diff <= matching_window_seconds and diff < min_diff:
                            min_diff = diff
                            best_match = peak
                    
                    # 如果找到了匹配项，则记录并从可用列表中移除
                    if best_match is not None:
                        matched_y.append(ref_peak)
                        matched_X.append(best_match)
                        available_peaks.remove(best_match)

                if not tqdm_desc:
                    print(f"找到了 {len(matched_X)} 个有效匹配的同步点。")

                matched_ref_peaks = []
                matched_target_peaks = []

                if len(matched_X) < 2:
                    print("警告: 有效匹配的同步点过少(<2)，只能进行简单首个对齐，无法计算漂移。")
                    offset = reference_peaks[0] - peaks[0] if len(reference_peaks) > 0 and len(peaks) > 0 else 0
                    drift = 1.0
                    # Use the raw matches if available
                    matched_ref_peaks = matched_y
                    matched_target_peaks = matched_X
                else:
                    X = np.array(matched_X).reshape(-1, 1)
                    y = np.array(matched_y)
                    ransac = RANSACRegressor()
                    ransac.fit(X, y)

                    # --- RANSAC 匹配结果可视化 ---
                    if visualize and len(matched_X) >= 2:
                        try:
                            inlier_mask = ransac.inlier_mask_
                            outlier_mask = np.logical_not(inlier_mask)
                            
                            # 创建用于绘制拟合线的点
                            line_X = np.arange(X.min(), X.max())[:, np.newaxis]
                            line_y_ransac = ransac.predict(line_X)

                            plt.figure(figsize=(10, 8))
                            # 绘制内点
                            plt.scatter(X[inlier_mask], y[inlier_mask], color='blue', marker='o', label='Inliers')
                            # 绘制离群点
                            plt.scatter(X[outlier_mask], y[outlier_mask], color='red', marker='x', label='Outliers')
                            # 绘制RANSAC拟合线
                            plt.plot(line_X, line_y_ransac, color='green', linestyle='-', linewidth=2, label='RANSAC Regressor')
                            
                            ref_video_name = os.path.basename(video_paths[0])
                            current_video_name = os.path.basename(v_path)
                            plt.title(f"RANSAC Matched Points: {ref_video_name} vs {current_video_name}")
                            plt.xlabel(f"Detected Time (s) in {current_video_name}")
                            plt.ylabel(f"Reference Time (s) in {ref_video_name}")
                            plt.legend()
                            plt.grid(True)
                            plt.axis('equal') # 保持x,y轴尺度一致，更好地观察漂移
                            output_filename = f"ransac_match_{ref_video_name}_vs_{current_video_name}.png"
                            plt.savefig(output_filename)
                            plt.close() 
                            if not tqdm_desc:
                                print(f"已将 RANSAC 匹配图表保存为 {output_filename}")
                        except (AttributeError, ValueError) as e:
                            # 如果RANSAC没有成功(e.g., no inliers), inlier_mask_ 会不存在
                            print(f"无法创建匹配可视化图表: {e}")
                        except Exception as e:
                            print(f"创建匹配可视化时发生未知错误: {e}")


                    if ransac.estimator_ is not None and hasattr(ransac.estimator_, 'coef_'):
                        drift = ransac.estimator_.coef_[0]
                        offset = ransac.estimator_.intercept_
                        
                        # Get inlier points for frame extraction
                        inlier_mask = ransac.inlier_mask_
                        inlier_ref_peaks = y[inlier_mask]
                        inlier_target_peaks = X[inlier_mask].flatten()
                        
                        # Sort by time to ensure intervals are sequential
                        sort_order = np.argsort(inlier_ref_peaks)
                        matched_ref_peaks = inlier_ref_peaks[sort_order].tolist()
                        matched_target_peaks = inlier_target_peaks[sort_order].tolist()
                    else:
                        print("警告: RANSAC 未能找到稳定的模型，退回到简单对齐。")
                        offset = reference_peaks[0] - peaks[0]
                        drift = 1.0
                        # Fallback to using raw matches
                        sort_order = np.argsort(matched_y)
                        matched_ref_peaks = np.array(matched_y)[sort_order].tolist()
                        matched_target_peaks = np.array(matched_X)[sort_order].tolist()

                results.append({
                    "file": v_path,
                    "offset_seconds": offset,
                    "drift_scale": drift,
                    "matched_points": {
                        "reference_peaks": matched_ref_peaks,
                        "target_peaks": matched_target_peaks
                    }
                })
                if not tqdm_desc:
                    print(f"结果 -> 偏移: {offset:.4f}秒, 速率修正: {drift:.6f}")

        if not results:
            return []

        # --- 新增：生成基于参考视频帧率的同步时间戳 ---
        if not tqdm_desc:
            print("\n" + "-"*20 + " 生成同步时间戳 " + "-"*20)

        # 1. 获取参考视频的FPS
        ref_video_path = results[0]['file']
        ref_fps = self._get_video_fps(ref_video_path)
        if ref_fps is None:
            print(f"[错误] 无法获取参考视频 {os.path.basename(ref_video_path)} 的FPS，无法生成同步时间戳。")
            # 为每个结果添加一个空列表，以保持数据结构一致性
            for res in results:
                res['synchronized_timestamps'] = []
            return results

        if not tqdm_desc:
            print(f"使用参考视频FPS: {ref_fps:.2f} 作为基准。")

        # 2. 遍历所有视频（包括参考视频）以生成时间戳
        for result in results:
            all_sync_ts = []
            ref_peaks = result['matched_points']['reference_peaks']
            target_peaks = result['matched_points']['target_peaks']

            if len(ref_peaks) < 2:
                print(f"警告: 视频 {os.path.basename(result['file'])} 匹配点少于2个，无法生成区间时间戳。")
                result['synchronized_timestamps'] = []
                continue

            # 3. 遍历每个同步信号之间的区间
            for i in range(len(ref_peaks) - 1):
                ref_interval_duration = ref_peaks[i+1] - ref_peaks[i]
                num_frames_in_interval = int(round(ref_interval_duration * ref_fps))
                
                if num_frames_in_interval <= 0: continue
                
                interval_timestamps = np.linspace(target_peaks[i], target_peaks[i+1], num_frames_in_interval, endpoint=False)
                all_sync_ts.extend(interval_timestamps.tolist())

            if len(target_peaks) > 0: all_sync_ts.append(target_peaks[-1])
            result['synchronized_timestamps'] = all_sync_ts
            if not tqdm_desc:
                print(f"为 {os.path.basename(result['file'])} 生成了 {len(all_sync_ts)} 个同步时间点。")

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于音频Chirp信号的多机位视频同步工具")
    subparsers = parser.add_subparsers(dest="command", required=True, help="可用命令")

    # --- 'generate-sound' 命令 ---
    parser_gen = subparsers.add_parser("generate-sound", help="生成用于现场播放的同步音频文件")
    parser_gen.add_argument("-o", "--output", default="sync_chirp_loop.wav", help="输出的 .wav 文件名")
    parser_gen.add_argument("--duration", type=float, default=0.3, help="Chirp信号持续时间(秒)")
    parser_gen.add_argument("--interval", type=float, default=10.0, help="两次信号之间的静音间隔(秒)")
    parser_gen.add_argument("--start-freq", type=int, default=2000, help="起始频率(Hz)")
    parser_gen.add_argument("--end-freq", type=int, default=6000, help="终止频率(Hz)")
    parser_gen.add_argument("--rate", type=int, default=48000, help="采样率(Hz)")

    # --- 'align' 命令 ---
    parser_align = subparsers.add_parser("align", help="分析视频文件并计算对齐参数")
    parser_align.add_argument("video_dir", help="包含视频文件的目录")
    parser_align.add_argument("--visualize", action="store_true", help="生成并保存峰值检测图表")
    parser_align.add_argument("--json-output", help="可选：将对齐结果保存为 JSON 文件路径")
    # 保留与 generate-sound 相同的参数，以确保分析时使用相同的Chirp配置
    parser_align.add_argument("--duration", type=float, default=0.3, help="Chirp信号持续时间(秒)")
    parser_align.add_argument("--start-freq", type=int, default=2000, help="起始频率(Hz)")
    parser_align.add_argument("--end-freq", type=int, default=6000, help="终止频率(Hz)")
    parser_align.add_argument("--rate", type=int, default=48000, help="采样率(Hz)")
    parser_align.add_argument("--window", type=float, default=3.0, help="匹配窗口大小(秒)")

    args = parser.parse_args()

    # 初始化同步系统
    syncer = AudioSyncSystem(
        chirp_duration=args.duration,
        start_freq=args.start_freq,
        end_freq=args.end_freq,
        sample_rate=args.rate
    )

    if args.command == "generate-sound":
        syncer.save_reference_audio(args.output, interval_silence=args.interval)
    
    elif args.command == "align":
        if not os.path.isdir(args.video_dir):
            print(f"错误: 提供的路径不是一个目录: {args.video_dir}")
            sys.exit(1)

        video_extensions = ['.mov', '.mp4', '.avi', '.m4v']
        video_files = []
        for f in sorted(os.listdir(args.video_dir)):
            if os.path.splitext(f)[1].lower() in video_extensions:
                video_files.append(os.path.join(args.video_dir, f))
        
        if not video_files:
            print(f"错误: 在目录 {args.video_dir} 中未找到支持的视频文件。")
            sys.exit(1)

        print(f"在目录中找到 {len(video_files)} 个视频文件进行处理。")
        alignment_results = syncer.align_videos(video_files, matching_window_seconds=args.window, visualize=args.visualize)
        
        if alignment_results:
            # 如果指定了 JSON 输出路径，则保存文件
            if args.json_output:
                try:
                    with open(args.json_output, 'w') as f:
                        json.dump(alignment_results, f, indent=4)
                    print(f"\n[成功] 对齐结果已保存至: {args.json_output}")
                except Exception as e:
                    print(f"\n[错误] 无法保存 JSON 文件: {e}")

            print("\n" + "="*20 + " 对齐结果 " + "="*20)
            for result in alignment_results:
                file_name = os.path.basename(result['file'])
                offset = result.get('offset_seconds', 0.0)
                drift = result.get('drift_scale', 1.0)
                
                print(f"\n文件: {file_name}")
                print(f"  - 时间偏移 (Offset): {offset:.4f} 秒")
                print(f"  - 速率修正 (Drift): {drift:.6f}")
                
                # 打印前几个原始同步点作为示例
                matched_pts = result.get('matched_points', {})
                raw_ts = matched_pts.get('target_peaks', [])
                
                if raw_ts:
                    print(f"  - 原始同步点 (前3个):   {[f'{t:.4f}s' for t in raw_ts[:3]]} ...")
                
                sync_ts = result.get('synchronized_timestamps', [])
                if sync_ts:
                    print(f"  - 均分后同步点 (前3个): {[f'{t:.4f}s' for t in sync_ts[:3]]} ... (共 {len(sync_ts)} 个)")

            print("\n" + "="*52)
            print("提示: '时间偏移'为负数表示该视频需要向前移动。'速率修正' > 1.0 表示该视频时钟比参考视频慢，需要轻微加速。")
