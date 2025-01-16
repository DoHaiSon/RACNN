from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, enable_logging=True, log_dir=None):
        self.enable_logging = enable_logging
        self.writer = SummaryWriter(log_dir=log_dir) if enable_logging else None
    
    def add_figure(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_figure(*args, **kwargs)
    
    def add_text(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_text(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_scalar(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_histogram(*args, **kwargs)
    
    def add_image(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_image(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_video(*args, **kwargs)
    
    def add_audio(self, *args, **kwargs):
        if self.enable_logging:
            self.writer.add_audio(*args, **kwargs)

    def flush(self):
        if self.enable_logging:
            self.writer.flush()

    def close(self):
        if self.enable_logging:
            self.writer.close()