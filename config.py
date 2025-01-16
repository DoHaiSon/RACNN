import yaml
import argparse
import os
from utils.print_utils import setup_print

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary with merged base config if specified
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # If base_config exists, load and merge with current config
    if 'base_config' in config:
        base_file = os.path.join('configs', f"{config['base_config']}.yaml")
        with open(base_file, 'r') as f:
            base_config = yaml.safe_load(f)
        # Merge base config with current config (current config takes precedence)
        base_config.update(config)
        config = base_config
    
    return config

def get_args():
    """
    Parse command line arguments and load YAML configuration.
    
    This function:
    1. Sets up argument parser
    2. Adds config file argument
    3. Loads YAML config
    4. Updates arguments with config values
    
    Returns:
        argparse.Namespace: Namespace containing all configuration parameters
    """
    parser = argparse.ArgumentParser()
    
    # Add argument for config file path
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Update args with values from config file
    for k, v in config.items():
        setattr(args, k, v)

    # Check verbose mode
    setup_print(args.verbose)
    
    return args