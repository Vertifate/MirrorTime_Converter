#!/usr/bin/env python3
"""
MirrorTime Converter - Simplified Launcher
#WDD [2026-01-20] [重构启动脚本：指向新的模块化简化版 GUI]
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # 路径配置
    base_dir = Path(__file__).parent
    gui_script = base_dir / "modules" / "gui" / "main.py"
    
    # 查找 Conda 环境中的 Python
    python_exe = sys.executable
    conda_env = Path.home() / "miniconda3" / "envs" / "mirrortime" / "bin" / "python"
    
    if conda_env.exists():
        python_exe = str(conda_env)
        print(f"[*] Environment: {python_exe}")
    
    if not gui_script.exists():
        print(f"Error: GUI script not found at {gui_script}")
        sys.exit(1)

    print("Launching MirrorTime Converter GUI...")
    try:
        # 启动后端 (集成前端静态服务)
        subprocess.run([python_exe, str(gui_script)], check=True)
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
