#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

"""
#WDD 2026-01-02 批量全流程脚本：训练 -> 清理 -> 生成 Mask
流程：
1. 遍历 base_dir 下的所有帧目录 (frameXXXX)。
2. 拷贝准备好的 sparse 文件夹到各帧目录。
3. 调用 example_train.py 进行训练。
4. 调用 clean_gs_by_cylinder.py 对生成的 PLY 进行清理（裁剪、去噪）。
5. 调用 generate_masks.py 根据清理后的 PLY 生成 Mask 和抠图。
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Full Pipeline Batch: Train -> Clean -> Mask")
    # 路径参数
    parser.add_argument("--base_dir", required=True, help="包含多个 frameXXXX 目录及 sparse 的根目录")
    parser.add_argument("--output_root", default=None, help="训练输出根目录 (默认: base_dir/tem)")
    parser.add_argument("--images_folder", default="images", help="帧目录下的图像子目录名")
    parser.add_argument("--sparse_dir", default=None, help="稀疏重建目录，默认 base_dir/sparse")
    
    # 训练控制
    parser.add_argument("--skip_train", action="store_true", help="跳过训练阶段")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--sh_degree", type=int, default=0)
    
    # 清理控制
    parser.add_argument("--skip_clean", action="store_true", help="跳过清理阶段")
    parser.add_argument("--clean_scale", type=float, default=0.75, help="清理裁剪半径比例")
    parser.add_argument("--camera_radius", type=float, default=0.1, help="相机避让半径比例")
    
    # 手动清理圆柱控制
    parser.add_argument("--clean_center_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="清理圆柱中心偏移 (x y z)")
    parser.add_argument("--clean_radius", type=float, default=-1.0, help="手动指定清理圆柱半径")
    parser.add_argument("--clean_height", type=float, default=-1.0, help="手动指定清理圆柱高度")
    
    # Mask 生成控制
    parser.add_argument("--skip_mask", action="store_true", help="跳过 Mask 生成阶段")
    parser.add_argument("--save_images_masked", action="store_true", help="是否保存带 Mask 的合成图像 (images_masked)")
    parser.add_argument("--save_images_color", action="store_true", help="是否保存彩色渲染图像 (images_color)")
    
    # 其他
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, help="传递给训练脚本的额外参数")
    
    return parser.parse_args()

def find_frames(base_dir: Path):
    pattern = re.compile(r"^frame_?\d+$")
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and pattern.match(p.name)])

def run_step(name, cmd, env=None):
    print(f"\n>>>> [Step: {name}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

def find_ply(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "point_cloud" / "finish" / "point_cloud.ply",
        output_dir / "point_cloud" / "point_cloud.ply",
        output_dir / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists(): return c
    matches = list(output_dir.glob("**/point_cloud.ply"))
    return matches[0] if matches else None

def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    
    base_dir = Path(args.base_dir).resolve()
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = base_dir / "tem"
        
    print(f"Output Root: {output_root}")
    
    sparse_src = Path(args.sparse_dir).resolve() if args.sparse_dir else base_dir / "sparse"
    
    #WDD 2026-01-02 默认在输出根目录下汇总
    collect_dir = output_root / "collect_cleaned"
    collect_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    
    # Add project root to PYTHONPATH so subprocesses can find 'litegs'
    project_root = str(root_dir)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_pythonpath}"
    
    # Set default extra_train_args if not provided
    if not args.extra_train_args:
        args.extra_train_args = ["-r", "4"]
    
    # Extract resolution from extra_train_args for mask generation
    train_resolution = 1  # default
    for i, arg in enumerate(args.extra_train_args):
        if arg == "-r" and i + 1 < len(args.extra_train_args):
            try:
                train_resolution = int(args.extra_train_args[i + 1])
            except ValueError:
                pass
            break

    frames = find_frames(base_dir)
    print(f"Found {len(frames)} frames in {base_dir}")
    print(f"Training resolution: -r {train_resolution}")

    for frame_dir in frames:
        frame_name = frame_dir.name
        print(f"\n{'='*20} Processing {frame_name} {'='*20}")
        
        frame_output_dir = output_root / frame_name
        
        # 1. Prepare sparse
        if not args.skip_train:
            dst_sparse = frame_dir / "sparse"
            if not dst_sparse.exists():
                print(f"[Prep] Copying sparse to {dst_sparse}")
                shutil.copytree(sparse_src, dst_sparse)
        
        # 2. Train
        if not args.skip_train:
            train_cmd = [
                sys.executable, str(root_dir / "example_train.py"),
                "-s", str(frame_dir),
                "-i", args.images_folder,
                "-m", str(frame_output_dir),
                "--iterations", str(args.iterations),
                "--sh_degree", str(args.sh_degree)
            ]
            if args.extra_train_args: train_cmd.extend(args.extra_train_args)
            run_step(f"Train {frame_name}", train_cmd, env)
        
        # 查找生成的 PLY
        ply_raw = find_ply(frame_output_dir)
        if not ply_raw:
            print(f"[Error] No PLY found for {frame_name} at {frame_output_dir}")
            continue
            
        ply_cleaned = ply_raw.parent / "point_cloud_cleaned.ply"
        
        # 3. Clean
        if not args.skip_clean:
            clean_cmd = [
                sys.executable, str(script_dir / "clean_gs_by_cylinder.py"),
                "-s", str(frame_dir),
                "-i", args.images_folder,
                "--input_ply", str(ply_raw),
                "--output_ply", str(ply_cleaned),
                "--scale", str(args.clean_scale),
                "--camera_radius", str(args.camera_radius),
                "--sh_degree", str(args.sh_degree),
                "--clean", # 启用 artifact 过滤
                "--cylinder_radius", str(args.clean_radius),
                "--cylinder_height", str(args.clean_height)
            ]
            # Pass offset
            clean_cmd.extend(["--center_offset"] + [str(x) for x in args.clean_center_offset])
            
            run_step(f"Clean {frame_name}", clean_cmd, env)
        else:
            # 如果跳过清理，则后续使用原始 PLY
            if not ply_cleaned.exists(): 
                ply_cleaned = ply_raw

        # 3.5. Collect cleaned PLY
        if collect_dir and ply_cleaned.exists():
            match = re.search(r"frame_?(\d+)", frame_name)
            suffix = match.group(1) if match else frame_name
            dst = collect_dir / f"{suffix}.ply"
            shutil.copy2(ply_cleaned, dst)
            print(f"[Collect] {ply_cleaned} -> {dst}")

        # 4. Generate Mask
        if not args.skip_mask:
            mask_cmd = [
                sys.executable, str(script_dir / "generate_masks.py"),
                "--source_path", str(frame_dir),
                "--image_dir", args.images_folder,
                "--input_ply", str(ply_cleaned),
                "-r", str(train_resolution)  # Pass resolution to match training
            ]
            if args.save_images_masked:
                mask_cmd.append("--save_images_masked")
            if args.save_images_color:
                mask_cmd.append("--save_images_color")
            
            run_step(f"MaskGen {frame_name}", mask_cmd, env)
            
            # Post-process cleanup is no longer needed as generate_masks.py handles conditional generation
    
    print("\n[All Done] Pipeline completed for all frames.")

if __name__ == "__main__":
    main()
