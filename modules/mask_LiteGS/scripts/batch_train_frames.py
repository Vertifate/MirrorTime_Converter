#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

"""
批量训练指定目录下的所有 frameXXXX 目录。
流程：
1. 找到 base_dir 下名称匹配 frame_XXXX 或 frameXXXX 的子目录。
2. 将 base_dir/sparse 整体拷贝到每个帧目录下（覆盖已有 sparse）。
3. 对每个帧调用 example_train.py 训练，输出到 output_root/<frame_name>。
4. 汇总所有生成的 point_cloud.ply 到 collect_dir，并按帧号命名为 <XXXX>.ply。

默认训练参数与单帧一致，可通过命令行调整。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Batch train LiteGS over frame directories")
    parser.add_argument("--base_dir", required=True, help="包含多个 frameXXXX 目录及 sparse 的根目录")
    parser.add_argument("--output_root", required=True, help="训练输出根目录")
    parser.add_argument("--collect_dir", required=True, help="汇总 point_cloud.ply 的目录")
    parser.add_argument("--images_folder", default="images", help="帧目录下的图像子目录名")
    parser.add_argument("--sparse_dir", default=None, help="稀疏重建目录，默认 base_dir/sparse")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=-1, help="输入分辨率（-1 表示不缩放）")
    parser.add_argument("--target_primitives", type=int, default=2_000_000)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--position_lr_max_steps", type=int, default=20_000)
    parser.add_argument("--position_lr_final", type=float, default=1.6e-5)
    parser.add_argument("--densification_interval", type=int, default=2)
    parser.add_argument("--eval", action="store_true", help="启用评测拆分")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="透传给 example_train.py 的额外参数")
    return parser.parse_args()


def find_frames(base_dir: Path):
    pattern = re.compile(r"^frame_?\d+$")
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and pattern.match(p.name)])


def ensure_sparse(frame_dir: Path, sparse_src: Path):
    dst = frame_dir / "sparse"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(sparse_src, dst)


def run_train(frame_dir: Path, output_root: Path, args):
    frame_name = frame_dir.name
    output_dir = output_root / frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "example_train.py"),
        "--sh_degree", str(args.sh_degree),
        "-s", str(frame_dir),
        "-i", args.images_folder,
        "-m", str(output_dir),
        "--resolution", str(args.resolution),
        "--target_primitives", str(args.target_primitives),
        "--iterations", str(args.iterations),
        "--position_lr_max_steps", str(args.position_lr_max_steps),
        "--position_lr_final", str(args.position_lr_final),
        "--densification_interval", str(args.densification_interval),
    ]
    if args.eval:
        cmd.append("--eval")
    if args.extra_args:
        cmd.extend(args.extra_args)

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    print(f"[Train] {frame_name}: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

    return output_dir


def find_ply(output_dir: Path) -> Path | None:
    """根据常见输出位置查找 point_cloud ply"""
    candidates = [
        output_dir / "point_cloud.ply",
        output_dir / "point_cloud" / "point_cloud.ply",
        output_dir / "point_cloud" / "finish" / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists():
            return c
    # 兜底：递归搜索 point_cloud*.ply
    matches = list(output_dir.glob("**/point_cloud*.ply"))
    return matches[0] if matches else None


def collect_ply(ply_path: Path, collect_dir: Path, frame_name: str):
    collect_dir.mkdir(parents=True, exist_ok=True)
    match = re.search(r"frame_?(\d+)", frame_name)
    suffix = match.group(1) if match else frame_name
    dst = collect_dir / f"{suffix}.ply"
    shutil.copy2(ply_path, dst)
    print(f"[Collect] {ply_path} -> {dst}")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    collect_dir = Path(args.collect_dir).expanduser().resolve()
    sparse_src = Path(args.sparse_dir).expanduser().resolve() if args.sparse_dir else base_dir / "sparse"

    if not sparse_src.exists():
        raise FileNotFoundError(f"sparse 源目录不存在: {sparse_src}")

    frames = find_frames(base_dir)
    if not frames:
        raise RuntimeError(f"未找到 frame 目录（frameXXXX 或 frame_XXXX）于 {base_dir}")

    for frame_dir in frames:
        print(f"[Prep] {frame_dir.name}: 拷贝 sparse -> {frame_dir/'sparse'}")
        ensure_sparse(frame_dir, sparse_src)

        output_dir = run_train(frame_dir, output_root, args)
        ply_path = find_ply(output_dir)
        if ply_path and ply_path.exists():
            collect_ply(ply_path, collect_dir, frame_dir.name)
        else:
            print(f"[Warn] 未找到 point_cloud ply 于 {output_dir}，跳过汇总")


if __name__ == "__main__":
    main()
