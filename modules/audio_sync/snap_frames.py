import ffmpeg
import json
import os
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def _get_raw_timestamps(video_path):
    """
    使用 ffprobe 快速获取视频每一帧的原始 PTS 时间戳。
    此函数不解码图像，只读取元数据，速度非常快。
    """
    try:
        probe = ffmpeg.probe(
            video_path,
            select_streams='v:0',
            show_entries='frame=pkt_pts_time'
        )
        
        timestamps = [float(frame['pkt_pts_time']) for frame in probe.get('frames', []) if 'pkt_pts_time' in frame]
        return timestamps
    except ffmpeg.Error as e:
        print(f"[FFmpeg 错误] 无法读取 {os.path.basename(video_path)}: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        print(f"[未知错误] 在处理 {os.path.basename(video_path)} 时发生: {e}")
        return None

class FrameSnapper:
    """
    将理想的时间戳列表“吸附”到每个视频最近的真实帧时间上。
    """
    def __init__(self, alignment_data, max_workers=os.cpu_count()):
        """
        初始化 FrameSnapper。
        
        :param alignment_data: 从 alignment_results.json 加载的数据。
        :param max_workers: 用于并行扫描视频的线程数。
        """
        self.alignment_data = alignment_data
        self.video_frame_maps = {}
        print("\n" + "="*20 + " 预加载视频元数据 " + "="*20)
        self._preload_all_frame_maps(max_workers)

    def _preload_all_frame_maps(self, max_workers):
        """
        使用多线程并行扫描所有视频，获取并缓存它们的真实帧时间戳。
        """
        video_paths = list(set([item['file'] for item in self.alignment_data if 'file' in item]))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_get_raw_timestamps, path): path for path in video_paths}
            
            with tqdm(total=len(future_to_path), desc="[预加载] 扫描视频帧时间戳") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        timestamps = future.result()
                        if timestamps is not None and len(timestamps) > 0:
                            self.video_frame_maps[path] = np.array(timestamps)
                        else:
                            print(f"\n警告: 未能从 {os.path.basename(path)} 获取有效的时间戳。")
                    except Exception as exc:
                        print(f"\n为 {os.path.basename(path)} 生成帧时间地图时出错: {exc}")
                    pbar.update(1)

    def _snap_timestamps(self, ideal_timestamps, real_timestamps):
        """
        为一组理想时间戳找到最近的真实时间戳。
        """
        if real_timestamps is None or len(real_timestamps) == 0:
            return []
            
        ideal_timestamps_np = np.array(ideal_timestamps)
        # 使用 NumPy 的广播和 argmin 功能进行高效的最近邻搜索
        # 这比在 Python 中循环快得多
        diffs = np.abs(real_timestamps[:, np.newaxis] - ideal_timestamps_np)
        indices = np.argmin(diffs, axis=0)
        snapped_timestamps = real_timestamps[indices]
        
        return snapped_timestamps.tolist()

    def snap_all_videos(self):
        """
        为所有视频执行时间戳对齐操作。
        :return: 一个字典，包含每个视频的理想时间、对齐后的真实时间和原始文件路径。
        """
        all_snapped_results = {}
        
        print("\n" + "="*20 + " 开始对齐到最近的真实帧 " + "="*20)
        
        for result in tqdm(self.alignment_data, desc="[对齐] 处理视频"):
            video_path = result.get('file')
            if not video_path:
                continue

            ideal_timestamps = result.get('synchronized_timestamps', [])
            real_timestamps = self.video_frame_maps.get(video_path)
            
            if not ideal_timestamps:
                print(f"警告: 视频 {os.path.basename(video_path)} 没有 'synchronized_timestamps'，已跳过。")
                snapped_ts = []
            elif real_timestamps is None:
                print(f"警告: 视频 {os.path.basename(video_path)} 没有可用的真实帧时间地图，已跳过。")
                snapped_ts = []
            else:
                snapped_ts = self._snap_timestamps(ideal_timestamps, real_timestamps)

            # 存储详细结果
            all_snapped_results[os.path.basename(video_path)] = {
                'file_path': video_path,
                'snapped_timestamps': snapped_ts,
                'mapping': [
                    {'ideal_time': ideal, 'snapped_time': snapped} 
                    for ideal, snapped in zip(ideal_timestamps, snapped_ts)
                ]
            }
        
        return all_snapped_results

def main():
    parser = argparse.ArgumentParser(description="将理想同步时间点对齐到视频的最近真实帧上。")
    parser.add_argument("json_file", help="包含 'synchronized_timestamps' 的 alignment_results.json 文件路径。")
    parser.add_argument("--output", help="可选：保存详细映射结果的 JSON 文件路径。")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="用于并行扫描视频的线程数。")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"错误: 找不到对齐文件 {args.json_file}")
        return

    try:
        with open(args.json_file, 'r') as f:
            alignment_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"错误: 读取或解析JSON文件失败: {e}")
        return

    if not alignment_data:
        print("错误: 对齐文件为空。")
        return

    # 1. 初始化 FrameSnapper，这将触发视频扫描
    snapper = FrameSnapper(alignment_data, max_workers=args.workers)
    
    # 2. 执行对齐计算
    snapped_results = snapper.snap_all_videos()

    # 3. 在控制台打印结果摘要
    print("\n" + "="*60)
    print("对齐完成！以下是每个视频前 3 个理想时间点及其对应的真实帧时间：")
    for vid_name, data in snapped_results.items():
        print(f"\n视频: {vid_name} (共找到 {len(data['snapped_timestamps'])} 个对齐时间点)")
        if not data['mapping']:
            print("  -> 无可用的时间点进行对齐。")
            continue
        for i in range(min(3, len(data['mapping']))):
            mapping = data['mapping'][i]
            ideal_t = mapping['ideal_time']
            snapped_t = mapping['snapped_time']
            diff_ms = (snapped_t - ideal_t) * 1000
            print(f"  理想时间 {ideal_t:.4f}s -> 对齐到真实帧 {snapped_t:.4f}s (误差: {diff_ms:+.2f}ms)")
    print("="*60)

    # 4. 如果指定，则保存完整的 JSON 输出
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(snapped_results, f, indent=4)
            print(f"\n[成功] 完整的对齐映射数据已保存至: {args.output}")
        except Exception as e:
            print(f"\n[错误] 保存文件失败: {e}")

if __name__ == "__main__":
    main()
