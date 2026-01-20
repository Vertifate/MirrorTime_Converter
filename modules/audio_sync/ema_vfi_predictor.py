import os
import sys
import os.path as osp
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 配置路径以包含模型定义
# ------------------------------------------------------------------------------
CURRENT_DIR = osp.dirname(osp.abspath(__file__))
MODEL_DIR = osp.join(CURRENT_DIR, 'InterpAny-Clearer', 'models', 'DI-EMA-VFI')

if not osp.exists(MODEL_DIR):
    MODEL_DIR = '/home/wyk/code/InterpAny-Clearer/models/DI-EMA-VFI'

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

# ------------------------------------------------------------------------------
# 导入模型依赖
# ------------------------------------------------------------------------------
try:
    from Trainer_recur import Model
    from benchmark.utils.padder import InputPadder
except ImportError as e:
    raise ImportError(f"无法从 {MODEL_DIR} 导入模型模块。错误信息: {e}")

class EMAVFIPredictor:
    def __init__(self, checkpoint_dir, iters=3, device='cuda'):
        """
        初始化 EMA-VFI 预测器。
        """
        # 1. 解析并设定设备
        if device == 'cpu':
            self.device = torch.device("cpu")
        else:
            # 如果传入的是 "cuda:7"，解析出 ID 7
            if ":" in device:
                try:
                    dev_id = int(device.split(":")[-1])
                    torch.cuda.set_device(dev_id) # [关键] 强制设定当前上下文
                except:
                    pass
            self.device = torch.device(device)

        self.iters = iters
        
        # 初始化模型结构 (Trainer_recur 会调用 self.device() -> self.net.to("cuda"))
        # 因为我们上面执行了 set_device，这里的 "cuda" 现在指的是我们指定的卡
        self.model = Model(-1)
        
        if not osp.exists(checkpoint_dir):
            alt_path = osp.normpath(osp.join(MODEL_DIR, checkpoint_dir))
            if osp.exists(alt_path):
                checkpoint_dir = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        if osp.isfile(checkpoint_dir):
            checkpoint_dir = osp.dirname(checkpoint_dir)
            
        self.model.load_model(log_path=checkpoint_dir)
        self.model.eval()
        
        # 再次确保 net 在正确的设备上 (双重保险)
        if hasattr(self.model, 'net'):
            self.model.net.to(self.device)

    @torch.no_grad()
    def predict(self, img0, img1, time_step):
        """
        在 img0 和 img1 之间生成指定时间步 time_step 的插帧图像。
        """
        # 预处理图像：转置并归一化
        # [优化] 确保 tensor 直接创建在目标 device 上，减少 CPU->GPU 传输开销
        img0_tensor = torch.from_numpy(img0.transpose(2, 0, 1)).float() / 255.
        img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float() / 255.
        I0 = img0_tensor.unsqueeze(0).to(self.device, non_blocking=True)
        I1 = img1_tensor.unsqueeze(0).to(self.device, non_blocking=True)
        
        padder = InputPadder(I0.shape, 32)
        I0, I1 = padder.pad(I0, I1)

        # 创建时间步 Embedding
        embt = torch.zeros_like(I0[:, :1, :, :]) + time_step

        mid = I0
        pre_mid = I0
        
        for j in range(self.iters):
            pre_embt = (embt * float(j)) / float(self.iters)
            cur_embt = (embt * (float(j) + 1)) / float(self.iters)
            
            mid = self.model.inference(I0, I1, pre_mid, cur_embt, pre_embt)
            pre_mid = mid
        
        mid = padder.unpad(mid)
        # 此时仍在 GPU 上，最后再转回 CPU
        mid = mid.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        
        return mid

if __name__ == '__main__':
    from argparse import ArgumentParser
    import cv2

    parser = ArgumentParser()
    parser.add_argument('--img0', type=str, required=True, help='Path to start image')
    parser.add_argument('--img1', type=str, required=True, help='Path to end image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save result')
    parser.add_argument('--iters', type=int, default=3, help='Recursion iterations')
    parser.add_argument('--time_step', type=float, default=0.5, help='Time step (0-1)')
    args = parser.parse_args()

    predictor = EMAVFIPredictor(checkpoint_dir=args.checkpoint, iters=args.iters)
    img0 = cv2.imread(args.img0)
    img1 = cv2.imread(args.img1)
    result = predictor.predict(img0, img1, args.time_step)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, 'result_{:.02f}.png'.format(args.time_step))
    cv2.imwrite(save_path, result)
    print('Interpolated frame saved to {}'.format(save_path))