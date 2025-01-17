# RACNN: Near-field Channel Estimation for 6G using Residual Attention CNN

RACNN (Residual Attention CNN) is a deep learning model designed for near-field channel estimation in 6G wireless communications. The model implements a novel architecture combining residual networks with attention mechanisms to improve channel estimation accuracy.

## Environment Setup
### Requirements
- Python 3.8+
- TensorFlow 2.10.0
- MATLAB (for data generation)
- NumPy
- SciPy
- PyYAML

### Installation
```bash
git clone https://github.com/DoHaiSon/RACNN.git
cd RACNN
conda env create -f env.yml
```

### Project Structure 
```
RACNN/
├── configs/                 # Configuration files
│   ├── defaults.yaml        # Default configuration
│   └── ...                  # Other config variants
├── data/                    # Dataset storage
├── gen_data/                # Data generation scripts
│   ├── noisy_channel.py     # Channel data generation
│   └── ...
├── models/                  # Model architectures
│   ├── CNN.py               # CNN implementation
│   └── RACNN.py             # RACNN implementation
├── utils/                   # Utility functions
├── main.py                  # Training script
└── test.py                  # Testing script
```

### Usage
#### Training
```bash
python main.py 
```

#### Testing
```bash
python test.py --ckpt path/to/model.keras
```

### Citation
If you use this code for your research, please cite:
```
@article{RACNN2024,
  title={RACNN: Near-field Channel Estimation for 6G using Residual Attention CNN},
  author={Do Hai Son},
  year={2024}
}
```

## Contribution

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.