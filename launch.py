#!/usr/bin/env python3
"""
MirrorTime Converter 主启动脚本
自动启动后端服务器和前端界面
"""
#WDD [2026-01-19] [创建主启动脚本]

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """打印启动横幅"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           🎬 MirrorTime Converter v1.0.0                  ║
║                                                           ║
║        4DGS 数据预处理可视化监控系统                        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
{Colors.ENDC}
    """
    print(banner)

def check_dependencies():
    """检查依赖是否安装"""
    print(f"{Colors.YELLOW}📦 检查依赖...{Colors.ENDC}")
    
    # 检查 Python 依赖
    backend_requirements = Path("modules/visualization-server/requirements.txt")
    if backend_requirements.exists():
        print(f"{Colors.BLUE}   检查后端依赖...{Colors.ENDC}")
        try:
            import fastapi
            import uvicorn
            print(f"{Colors.GREEN}   ✓ FastAPI 已安装{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.RED}   ✗ FastAPI 未安装{Colors.ENDC}")
            print(f"{Colors.YELLOW}   正在安装后端依赖...{Colors.ENDC}")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(backend_requirements)
            ], check=True)
    
    # 检查 Node.js 依赖
    frontend_path = Path("visualization-ui")
    if frontend_path.exists():
        node_modules = frontend_path / "node_modules"
        if not node_modules.exists():
            print(f"{Colors.YELLOW}   正在安装前端依赖...{Colors.ENDC}")
            subprocess.run(
                ["npm", "install"], 
                cwd=str(frontend_path),
                check=True
            )
        print(f"{Colors.GREEN}   ✓ 前端依赖已安装{Colors.ENDC}")
    
    print(f"{Colors.GREEN}✅ 依赖检查完成{Colors.ENDC}\n")

def start_backend():
    """启动后端服务器"""
    print(f"{Colors.CYAN}🚀 启动后端服务器...{Colors.ENDC}")
    
    backend_script = Path("modules/visualization-server/src/main.py")
    
    if not backend_script.exists():
        print(f"{Colors.RED}❌ 错误: 找不到后端脚本 {backend_script}{Colors.ENDC}")
        sys.exit(1)
    
    # 启动 FastAPI 服务器
    backend_process = subprocess.Popen(
        [sys.executable, str(backend_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"{Colors.GREEN}   ✓ 后端服务器已启动 (PID: {backend_process.pid}){Colors.ENDC}")
    print(f"{Colors.BLUE}   📡 API: http://localhost:8000{Colors.ENDC}")
    print(f"{Colors.BLUE}   📡 WebSocket: ws://localhost:8000/ws{Colors.ENDC}")
    print(f"{Colors.BLUE}   📖 文档: http://localhost:8000/docs{Colors.ENDC}\n")
    
    return backend_process

def start_frontend():
    """启动前端开发服务器"""
    print(f"{Colors.CYAN}🎨 启动前端界面...{Colors.ENDC}")
    
    frontend_path = Path("visualization-ui")
    
    if not frontend_path.exists():
        print(f"{Colors.RED}❌ 错误: 找不到前端目录 {frontend_path}{Colors.ENDC}")
        sys.exit(1)
    
    # 启动 Vite 开发服务器
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"{Colors.GREEN}   ✓ 前端服务器已启动 (PID: {frontend_process.pid}){Colors.ENDC}")
    print(f"{Colors.BLUE}   🌐 界面: http://localhost:5173{Colors.ENDC}\n")
    
    return frontend_process

def main():
    """主函数"""
    print_banner()
    
    # 进程列表
    processes = []
    
    def signal_handler(sig, frame):
        """处理退出信号"""
        print(f"\n{Colors.YELLOW}⏸️  正在关闭服务...{Colors.ENDC}")
        for process in processes:
            process.terminate()
        print(f"{Colors.GREEN}✅ 所有服务已关闭{Colors.ENDC}")
        sys.exit(0)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 检查依赖
        check_dependencies()
        
        # 启动后端
        backend_process = start_backend()
        processes.append(backend_process)
        time.sleep(2)  # 等待后端启动
        
        # 启动前端
        frontend_process = start_frontend()
        processes.append(frontend_process)
        time.sleep(2)  # 等待前端启动
        
        print(f"{Colors.GREEN}{Colors.BOLD}✨ MirrorTime Converter 已启动！{Colors.ENDC}")
        print(f"{Colors.CYAN}   请打开浏览器访问: http://localhost:5173{Colors.ENDC}")
        print(f"{Colors.YELLOW}   按 Ctrl+C 退出{Colors.ENDC}\n")
        
        # 实时输出日志
        while True:
            # 检查进程是否还在运行
            if backend_process.poll() is not None:
                print(f"{Colors.RED}❌ 后端服务器已退出{Colors.ENDC}")
                break
            if frontend_process.poll() is not None:
                print(f"{Colors.RED}❌ 前端服务器已退出{Colors.ENDC}")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"{Colors.RED}❌ 启动失败: {e}{Colors.ENDC}")
        for process in processes:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
