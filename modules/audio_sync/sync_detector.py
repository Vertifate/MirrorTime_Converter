import numpy as np
import scipy.signal as signal
import ffmpeg
import os

class SingleVideoSyncDetector:
    def __init__(self, chirp_duration=0.3, start_freq=2000, end_freq=6000, sample_rate=48000):
        """
        初始化单视频同步检测器。
        参数默认值应与录制时播放的音频信号保持一致。
        
        :param chirp_duration: 扫频信号持续时间(秒)
        :param start_freq: 起始频率 (Hz)
        :param end_freq: 终止频率 (Hz)
        :param sample_rate: 采样率
        """
        self.sr = sample_rate
        self.chirp_duration = chirp_duration
        self.f0 = start_freq
        self.f1 = end_freq
        # 初始化时预先生成参考信号模板
        self.reference_chirp = self._generate_chirp()

    def _generate_chirp(self):
        """生成线性调频信号 (Chirp) 用于匹配"""
        t = np.linspace(0, self.chirp_duration, int(self.sr * self.chirp_duration))
        # 生成 chirp 信号
        w = signal.chirp(t, f0=self.f0, f1=self.f1, t1=self.chirp_duration, method='linear')
        # 加窗函数减少频谱泄露
        window = signal.windows.hann(len(w))
        return w * window

    def extract_audio_from_video(self, video_path):
        """使用 ffmpeg 从视频文件中提取音频序列"""
        if not os.path.exists(video_path):
            print(f"[错误] 文件不存在: {video_path}")
            return None
            
        try:
            # 使用 ffmpeg-python 读取音频流
            out_bytes, err = (
                ffmpeg
                .input(video_path)
                .output(
                    'pipe:',            # 输出到管道
                    format='f32le',      # 格式: 32位浮点数
                    acodec='pcm_f32le',  # 编码: PCM
                    ac=1,                # 单声道
                    ar=self.sr           # 采样率
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
            # 转换为 NumPy 数组
            audio_data = np.frombuffer(out_bytes, np.float32)
            return audio_data
        except ffmpeg.Error as e:
            print(f"[FFmpeg 错误] 无法读取音频: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            print(f"[未知错误] {e}")
            return None

    def find_peaks(self, audio_data, threshold_percent=0.3):
        """
        核心算法：在音频数据中寻找同步信号峰值
        :return: (peak_times_seconds, correlation_signal)
        """
        # 1. 带通滤波：滤除环境噪音
        sos = signal.butter(10, [self.f0 * 0.9, self.f1 * 1.1], btype='bandpass', fs=self.sr, output='sos')
        filtered_audio = signal.sosfilt(sos, audio_data)

        # 2. 互相关计算 (Matched Filter)
        correlation = signal.fftconvolve(filtered_audio, self.reference_chirp[::-1], mode='full')
        correlation = np.abs(correlation)
        correlation /= np.max(correlation) if np.max(correlation) > 0 else 1

        # 3. 寻找峰值
        min_distance = int(self.sr * 1.0) # 假设信号间隔至少1秒
        peaks, _ = signal.find_peaks(correlation, height=threshold_percent, distance=min_distance)

        return peaks / self.sr, correlation

    def detect(self, video_path, threshold_percent=0.3):
        """
        主接口：检测视频中的同步信号
        :param video_path: 视频路径
        :param threshold_percent: 峰值检测阈值 (0.0 - 1.0)
        :return: 同步信号出现的时间点列表 (秒)
        """
        audio_data = self.extract_audio_from_video(video_path)
        if audio_data is None or len(audio_data) == 0:
            return []
        peak_times, _ = self.find_peaks(audio_data, threshold_percent)
        return peak_times.tolist()