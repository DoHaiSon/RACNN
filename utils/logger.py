import os
import datetime
import tensorflow as tf

class Logger:
    def __init__(self, enable_logging=True, default_log_dir=None):
        self.enable_logging = enable_logging
        self.writer = tf.summary.create_file_writer(default_log_dir) if enable_logging else None
    
    def add_figure(self, tag, figure, step=None):
        if self.enable_logging:
            with self.writer.as_default():
                tf.summary.image(tag, self._plot_to_image(figure), step=step)
    
    def add_text(self, tag, text, step=None):
        if self.enable_logging:
            with self.writer.as_default():
                tf.summary.text(tag, text, step=step)

    def add_scalar(self, tag, scalar_value, step=None):
        if self.enable_logging:
            with self.writer.as_default():
                tf.summary.scalar(tag, scalar_value, step=step)

    def add_histogram(self, tag, values, step=None, bins=None):
        if self.enable_logging:
            with self.writer.as_default():
                tf.summary.histogram(tag, values, step=step, buckets=bins)
    
    def add_image(self, tag, image, step=None):
        if self.enable_logging:
            with self.writer.as_default():
                tf.summary.image(tag, image, step=step)

    def add_audio(self, tag, audio, sample_rate, step=None):
        if self.enable_logging:
            with self.writer.as_default():
                tf.summary.audio(tag, audio, sample_rate, step=step)

    def flush(self):
        if self.enable_logging:
            self.writer.flush()

    def close(self):
        if self.enable_logging:
            self.writer.close()

    def _plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it."""
        import io
        from PIL import Image
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside the notebook.
        figure.clf()
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image