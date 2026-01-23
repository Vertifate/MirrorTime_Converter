from tqdm import tqdm

class ProgressTqdm(tqdm):
    """
    A wrapper around tqdm that triggers a callback on update.
    This allows syncing CLI progress bars with UI/WebSockets.
    """
    def __init__(self, iterable=Nodene, progress_callback=None, progress_scale=(0.0, 100.0), *args, **kwargs):
        """
        :param progress_callback: func(percent: float) -> void. 
                                  Note: Ensure this callback is thread-safe if used in threads.
        :param progress_scale: tuple (start_percent, end_percent) maps the tqdm 0-100% to this range.
                               Example: (50.0, 100.0) means 0% tqdm -> 50% global, 100% tqdm -> 100% global.
        """
        self.progress_callback = progress_callback
        self.scale_start = progress_scale[0]
        self.scale_width = progress_scale[1] - progress_scale[0]
        
        # Initialize standard tqdm
        super().__init__(iterable, *args, **kwargs)

    def update(self, n=1):
        """Overrides tqdm.update to trigger callback."""
        # Call original update logic
        super().update(n)
        self._trigger_callback()

    def _trigger_callback(self):
        if self.progress_callback and self.total:
            # Calculate raw completion ratio (0.0 to 1.0)
            # Use self.n (current items) / self.total (total items)
            raw_ratio = self.n / float(self.total)
            
            # Map raw ratio to the global scale
            scaled_percent = self.scale_start + (raw_ratio * self.scale_width)
            
            # Clamp percentage to ensure it stays within 0-100 logically
            scaled_percent = max(0.0, min(100.0, scaled_percent))
            
            # #WDD [2026-01-20] [修复] 传递两个参数以匹配新的回调签名
            # 第一个参数是描述信息，第二个参数是进度百分比
            desc_msg = f"{self.desc or 'Processing'}: {self.n}/{self.total}"
            self.progress_callback(desc_msg, scaled_percent)
