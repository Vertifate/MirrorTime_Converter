import os
import json
import argparse
import numpy as np
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import bisect

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
        return np.array(timestamps) # 返回 numpy 数组以便处理
    except ffmpeg.Error as e:
        print(f"[FFmpeg 错误] 无法读取 {os.path.basename(video_path)}: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        print(f"[未知错误] 在处理 {os.path.basename(video_path)} 时发生: {e}")
        return None

class ExtractionPlanner:
    def __init__(self, snapped_data, max_workers=os.cpu_count()):
        """
        初始化规划器。
        :param snapped_data: snap_frames.py 的输出数据 (字典格式)
        """
        self.snapped_data = snapped_data
        self.video_full_timestamps = {}
        
        # 提取所有视频路径
        self.video_paths = []
        for key, val in snapped_data.items():
            if isinstance(val, dict) and 'file_path' in val:
                self.video_paths.append(val['file_path'])
        
        # 去重
        self.video_paths = list(set(self.video_paths))
        
        print("\n" + "="*20 + " 正在扫描视频完整时间轴 (用于插帧计算) " + "="*20)
        self._preload_timestamps(max_workers)

    def _preload_timestamps(self, max_workers):
        """并行获取所有视频的完整时间戳列表"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_get_raw_timestamps, path): path for path in self.video_paths}
            
            with tqdm(total=len(future_to_path), desc="[扫描] 获取完整帧信息") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        timestamps = future.result()
                        if timestamps is not None and len(timestamps) > 0:
                            self.video_full_timestamps[path] = timestamps
                        else:
                            print(f"\n警告: 无法获取 {os.path.basename(path)} 的时间戳。")
                    except Exception as exc:
                        print(f"\n处理 {os.path.basename(path)} 时出错: {exc}")
                    pbar.update(1)

    def plan(self):
        """
        执行规划逻辑，生成详细的抽帧/插帧计划。
        """
        print("\n" + "="*20 + " 正在制定提取计划 " + "="*20)
        
        # 1. 数据重组 (Pivot): 从 "按视频" 转为 "按帧索引"
        # 我们假设所有视频的 mapping 长度一致（基于 synchronized_timestamps 的生成逻辑）
        # 先找到一个有效的视频作为基准来确定帧数
        num_frames = 0
        reference_video_key = None
        for vid_key, data in self.snapped_data.items():
            if 'mapping' in data and len(data['mapping']) > 0:
                num_frames = len(data['mapping'])
                reference_video_key = vid_key
                break
        
        if num_frames == 0:
            print("错误: 数据中没有有效的帧映射信息。")
            return {}

        extraction_plan = {}
        
        # 遍历每一帧 (同步时刻)
        for i in tqdm(range(num_frames), desc="[规划] 计算每一帧策略"):
            frame_key = f"frame_{i:06d}"
            
            # --- 第一步: 收集该时刻所有视频的误差 ---
            errors = []
            frame_video_infos = [] # 临时存储，避免重复查找
            
            for vid_name, data in self.snapped_data.items():
                if 'mapping' not in data or i >= len(data['mapping']):
                    continue
                
                mapping_item = data['mapping'][i]
                ideal_time = mapping_item['ideal_time']
                snapped_time = mapping_item['snapped_time'] # 这是最近的真实帧时间
                
                # 误差 = 真实 - 理想
                error = snapped_time - ideal_time
                errors.append(error)
                
                frame_video_infos.append({
                    'vid_name': vid_name,
                    'file_path': data['file_path'],
                    'ideal_time': ideal_time,
                    'snapped_time': snapped_time # 这里的 snapped_time 仅用于计算初始误差
                })
            
            if not errors:
                continue

            # --- 第二步: 计算中位数误差 ---
            median_shift = float(np.median(errors))
            
            # --- 第三步: 为每个视频制定决策 ---
            plan_for_frame = {
                "median_shift": median_shift,
                "videos": {}
            }
            
            for info in frame_video_infos:
                vid_name = info['vid_name']
                file_path = info['file_path']
                ideal_time = info['ideal_time']
                
                # 计算修正后的目标时间
                target_time = ideal_time + median_shift
                
                # 获取该视频的完整时间轴
                full_timestamps = self.video_full_timestamps.get(file_path)
                if full_timestamps is None:
                    # 如果没有完整时间轴，无法进行精确计算，回退到使用 snapped_time
                    # 这种情况通常是 ffprobe 失败，标记为错误或仅提取
                    plan_for_frame["videos"][vid_name] = {
                        "action": "error_missing_timestamps",
                        "ideal_time": ideal_time
                    }
                    continue

                # 在完整时间轴中找到离 target_time 最近的帧
                # 使用 bisect 找到插入位置
                idx = bisect.bisect_left(full_timestamps, target_time)
                
                # 确定最近的帧索引
                closest_idx = 0
                if idx == 0:
                    closest_idx = 0
                elif idx == len(full_timestamps):
                    closest_idx = len(full_timestamps) - 1
                else:
                    # 比较 idx 和 idx-1 哪个更近
                    before = full_timestamps[idx - 1]
                    after = full_timestamps[idx]
                    if (target_time - before) < (after - target_time):
                        closest_idx = idx - 1
                    else:
                        closest_idx = idx
                
                closest_real_time = float(full_timestamps[closest_idx])
                
                # 计算差距
                diff = abs(closest_real_time - target_time)
                
                # 决策阈值: 3ms = 0.003s
                if diff < 0.003:
                    # --- 方案 A: 直接抽帧 ---
                    plan_for_frame["videos"][vid_name] = {
                        "action": "extract",
                        "timestamp": closest_real_time,
                        "frame_idx": closest_idx,
                        "diff": diff,
                        "ideal_time_original": ideal_time,
                        "target_time": target_time
                    }
                else:
                    # --- 方案 B: 插帧 ---
                    # 我们需要找到 T_prev <= target_time < T_next
                    # bisect_right 返回的 idx 是第一个 > target_time 的位置
                    idx_right = bisect.bisect_right(full_timestamps, target_time)
                    
                    prev_idx = idx_right - 1
                    next_idx = idx_right
                    
                    # 边界检查
                    if prev_idx < 0:
                        # 目标时间在视频开始之前 -> 无法插帧，只能取第一帧
                        plan_for_frame["videos"][vid_name] = {
                            "action": "extract", # 强制回退到抽帧
                            "timestamp": float(full_timestamps[0]),
                            "frame_idx": 0,
                            "note": "out_of_bounds_start",
                            "ideal_time_original": ideal_time
                        }
                    elif next_idx >= len(full_timestamps):
                        # 目标时间在视频结束之后 -> 无法插帧，只能取最后一帧
                        plan_for_frame["videos"][vid_name] = {
                            "action": "extract", # 强制回退到抽帧
                            "timestamp": float(full_timestamps[-1]),
                            "frame_idx": len(full_timestamps) - 1,
                            "note": "out_of_bounds_end",
                            "ideal_time_original": ideal_time
                        }
                    else:
                        # 正常插帧情况
                        t_prev = float(full_timestamps[prev_idx])
                        t_next = float(full_timestamps[next_idx])
                        
                        # 计算插值比例 (0.0 - 1.0)
                        # time_step = (target - prev) / (next - prev)
                        denom = t_next - t_prev
                        if denom > 0:
                            interp_step = (target_time - t_prev) / denom
                        else:
                            interp_step = 0.0 # 避免除以零
                        
                        plan_for_frame["videos"][vid_name] = {
                            "action": "interpolate",
                            "target_time": target_time,
                            "prev_frame_time": t_prev,
                            "next_frame_time": t_next,
                            "prev_frame_idx": prev_idx,
                            "next_frame_idx": next_idx,
                            "interp_step": interp_step,
                            "diff": diff,
                            "ideal_time_original": ideal_time
                        }

            extraction_plan[frame_key] = plan_for_frame
            
        return extraction_plan

def main():
    parser = argparse.ArgumentParser(description="根据对齐数据制定详细的抽帧/插帧计划。")
    parser.add_argument("snapped_json", help="snap_frames.py 生成的 JSON 文件路径。")
    parser.add_argument("--output", help="可选：保存提取计划的 JSON 文件路径。")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="线程数。")
    args = parser.parse_args()

    if not os.path.exists(args.snapped_json):
        print(f"错误: 找不到文件 {args.snapped_json}")
        return

    # 1. 读取输入数据
    try:
        with open(args.snapped_json, 'r') as f:
            snapped_data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return

    # 2. 初始化规划器 (会触发视频扫描)
    planner = ExtractionPlanner(snapped_data, max_workers=args.workers)
    
    # 3. 制定计划
    plan = planner.plan()
    
    # 4. 统计信息
    total_frames = len(plan)
    total_actions = 0
    extract_count = 0
    interp_count = 0
    
    for frame_data in plan.values():
        for vid_data in frame_data['videos'].values():
            total_actions += 1
            if vid_data['action'] == 'extract':
                extract_count += 1
            elif vid_data['action'] == 'interpolate':
                interp_count += 1

    print("\n" + "="*20 + " 计划摘要 " + "="*20)
    print(f"总同步时刻数 (Frames): {total_frames}")
    print(f"总处理动作数 (Videos * Frames): {total_actions}")
    print(f"  - 直接抽帧 (Extract): {extract_count} ({extract_count/total_actions*100:.1f}%)")
    print(f"  - AI 插帧 (Interpolate): {interp_count} ({interp_count/total_actions*100:.1f}%)")
    
    # 5. 输出
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(plan, f, indent=4)
            print(f"\n[成功] 提取计划已保存至: {args.output}")
        except Exception as e:
            print(f"\n[错误] 保存文件失败: {e}")
    else:
        # 如果没有指定输出文件，打印前几帧的计划作为示例
        print("\n[提示] 未指定输出文件，仅打印前 2 帧的计划示例：")
        keys = sorted(list(plan.keys()))[:2]
        example_subset = {k: plan[k] for k in keys}
        print(json.dumps(example_subset, indent=2))

if __name__ == "__main__":
    main()
