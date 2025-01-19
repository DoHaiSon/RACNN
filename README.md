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
conda activate RACNN
```

### Project Structure 
```
RACNN/
├── configs/                 # Configuration files
│   ├── defaults.yaml        # Default configuration
│   └── ...                  # Other config variants
├── data/                    # Dataset storage
├── gen_data/                # Data generation scripts
│   ├── noisy_channel.py     # Channel data generation during training
│   └── gen_data.m           # MATLAB script for data generation
├── models/                  # Model architectures
│   ├── CNN.py               # CNN implementation
│   └── RACNN.py             # RACNN implementation
├── utils/                   # Utility functions
├── config.py                # Do not modify
├── train.py                 # Training script
├── test.py                  # Testing script
└── env.yml                  # Conda environment configuration
```

### Usage
#### Dataset Generation
Before training and testing the model, you need to generate the dataset using MATLAB.

1. Open MATLAB and navigate to the `gen_data` directory.
2. Run the `gen_data.m` script to generate the channel data.
3. The generated data will be saved in the `data` directory.

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
@inproceedings{RACNN2025,
  title={RACNN: Near-field Channel Estimation for 6G using Residual Attention CNN},
  author={Lam, Vu Tung and Son, Do Hai and Quynh, Tran Thi Thuy and Le, Trung Thanh},
  year={2025}
}
```

## Contribution

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.