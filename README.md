# UniTS
[![Ubuntu](https://img.shields.io/badge/Ubuntu-orange)](https://ubuntu.com/) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/) [![PyTorch 2.4.0+cu124](https://img.shields.io/badge/PyTorch-2.4.0%2Bcu124-red)](https://pytorch.org/) [![PyG 2.6.1](https://img.shields.io/badge/torch__geometric-2.6.1-green)](https://pytorch-geometric.readthedocs.io/) [![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-blueviolet)](https://www.rdkit.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of *"A unified framework for automated transition state generation enabling mechanistic exploration in organic synthesis"*. The corresponding paper is under review.

## 📌 Key Features
- Comprehensive Pipeline:​ UniTS combines the UniTS-Lib​ dataset with the UniTS-Gen​ model, offering a complete workflow from TS data to TS generation.
- General-Purpose Design:​ Built to handle the complexity of real-world synthetic chemistry, including organometallic catalysis and diverse reaction types.
- From 2D to 3D:​ The UniTS-Gen​ model generates accurate 3D TS geometries using only 2D molecular graphs and reactive site indices as input.

## ✨️ Showcase
Here, we display some of the generation TS trajatories (from Formula-OOS test set) from the UniTS-Gen model.

| OOS demo1 | OOS demo2 | OOS demo3 | OOS demo4 |
|:---------:|:---------:|:---------:|:---------:|
| <img src="gif/b5_s24.gif" width="200"> | <img src="gif/b8_s10.gif" width="200"> | <img src="gif/b11_s2.gif" width="200"> | <img src="gif/b11_s7.gif" width="200"> |

## 🚀 Quick Start
### Installation
```
conda create -n units python=3.11
conda activate units
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.4.0+cu124.html --extra-index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/licheng-xu-echo/QCBot.git
cd QCBot
pip install .
cd ..
git clone https://github.com/licheng-xu-echo/UniTS.git
cd UniTS
pip install .
```
**Note**: All codes were tested under Ubuntu 22.04.2 LTS

### Model weights and dataset
During the peer-review period, the model weights and the UniTS dataset are available upon request. Please contact us via [email](mailto:xulicheng@sais.org.cn) if you are interested.

## ⚛️ Model Training and Testing
```bash
# train model
python -u train.py --config_file ./config/train_hiegnn_unitslib_1GPU.json
```

```bash
# sample trajactory
python -u traj_sampling.py --model_root ./model_path --model_tag units_hiegnn --batch_size 32
```