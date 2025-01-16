import os
import subprocess

def find_latest_log_dir(log_dir):
    """
    Find the latest log directory in the given log_dir.
    """
    # Get the list of subdirectories in log_dir
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    # Sort the subdirectories by modification time and get the latest one
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def start_tensorboard(log_dir, port=6006):
    """
    Start TensorBoard with the given log_dir on the specified port.
    If the port is in use, try the next port (port + 1).
    """
    try:
        # Attempt to start TensorBoard on the specified port
        subprocess.run(["tensorboard", "--logdir", log_dir, "--port", str(port), '--load_fast', 'false'], check=True)
    except subprocess.CalledProcessError:
        # If the specified port is in use, try the next port
        subprocess.run(["tensorboard", "--logdir", log_dir, "--port", str(port + 1), '--load_fast', 'false'], check=True)

if __name__ == "__main__":
    # Define the log directory
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    # Find the latest log directory
    latest_log_dir = find_latest_log_dir(log_dir)
    print(f"Starting TensorBoard for the latest log directory: {latest_log_dir}")
    # Start TensorBoard for the latest log directory
    start_tensorboard(latest_log_dir)