#!/usr/bin/env python3
"""
诊断脚本：重现 MirrorTime 负值问题
"""

# 模拟 frame_000101 的计算
target_frame_idx = 101
ref_fps = 30.0

# 假设的 intersect_start (需要从实际运行中获取)
# 根据 offset 值，intersect_start 应该是 max(all_starts)
# 参考相机: global_start = 0.0
# 其他相机: global_start = offset (大部分为负)
# intersect_start = max(0.0, -0.1016, -0.0104, ...) = 0.0

intersect_start = 0.0  # 假设值

# 计算目标全局时间
target_time_relative = target_frame_idx / ref_fps
target_global_time = intersect_start + target_time_relative

print(f"=== Frame {target_frame_idx} 计算 ===")
print(f"intersect_start = {intersect_start:.4f}s")
print(f"target_time_relative = {target_time_relative:.4f}s")
print(f"target_global_time = {target_global_time:.4f}s")
print()

# 模拟几个相机的计算
cameras = [
    ("001.MOV", 0.0, 1.0),  # 参考相机
    ("004.MOV", -0.1096, 1.000000),  # 负 offset
    ("008.MOV", -0.1142, 1.000000),  # 更大的负 offset
]

print("=== 各相机计算结果 ===")
for fname, offset, drift in cameras:
    # 逆推本地时间
    local_time = (target_global_time - offset) / drift
    
    # 计算本地帧索引 (假设 fps=60)
    fps = 60.0
    local_frame_idx = int(local_time * fps)
    
    # 模拟读取 raw_pts
    # 如果 local_frame_idx < 0，则无法读取缓存，回退到估算
    if local_frame_idx < 0:
        raw_pts = local_frame_idx / fps  # 负值！
        print(f"\n{fname}:")
        print(f"  local_time = {local_time:.4f}s")
        print(f"  local_frame_idx = {local_frame_idx} ❌ NEGATIVE!")
        print(f"  raw_pts (fallback) = {raw_pts:.4f}s")
    else:
        raw_pts = local_frame_idx / fps
        print(f"\n{fname}:")
        print(f"  local_time = {local_time:.4f}s")
        print(f"  local_frame_idx = {local_frame_idx}")
        print(f"  raw_pts = {raw_pts:.4f}s")
    
    # 计算最终 MirrorTime
    abs_global = raw_pts * drift + offset
    final_time = abs_global
    
    print(f"  MirrorTime = {raw_pts:.4f} * {drift:.6f} + {offset:.4f} = {final_time:.4f}s")
    
    if final_time < 0:
        print(f"  ⚠️  NEGATIVE MirrorTime!")

print("\n=== 结论 ===")
print("当 offset 为负且较大时，早期帧的 local_time 会变成负数")
print("导致 local_frame_idx < 0，fallback 逻辑使用负的 raw_pts")
print("最终产生负的 MirrorTime")
