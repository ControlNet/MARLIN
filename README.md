# MARLIN: Masked Autoencoder for facial video Representation LearnINg

<div>
    <img src="assets/teaser.svg">
    <p></p>
</div>

<div align="center">
    <a href="https://github.com/ControlNet/MARLIN/network/members">
        <img src="https://img.shields.io/github/forks/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MARLIN/stargazers">
        <img src="https://img.shields.io/github/stars/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MARLIN/issues">
        <img src="https://img.shields.io/github/issues/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MARLIN/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2211.06627">
        <img src="https://img.shields.io/badge/arXiv-2211.06627-b31b1b.svg?style=flat-square">
    </a>
</div>

<div align="center">    
    <a href="https://pypi.org/project/marlin-pytorch/">
        <img src="https://img.shields.io/pypi/v/marlin-pytorch?style=flat-square">
    </a>
    <a href="https://pypi.org/project/marlin-pytorch/">
        <img src="https://img.shields.io/pypi/dm/marlin-pytorch?style=flat-square">
    </a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/marlin-pytorch?style=flat-square"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-EE4C2C?style=flat-square&logo=pytorch"></a>
    <a href="https://github.com/ControlNet/MARLIN/actions"><img src="https://img.shields.io/github/workflow/status/ControlNet/MARLIN/Unittest/dev?label=unittest&style=flat-square"></a>
    <a href="https://github.com/ControlNet/MARLIN/actions"><img src="https://img.shields.io/github/workflow/status/ControlNet/MARLIN/Release/master?label=release&style=flat-square"></a>
</div>

This repo is the official PyTorch implementation for the paper 
[MARLIN: Masked Autoencoder for facial video Representation LearnINg](https://arxiv.org/abs/2211.06627).

## Repository Structure

The repository contains 2 parts:
 - `marlin-pytorch`: The PyPI package for MARLIN used for inference.
 - The implementation for the paper including training and evaluation scripts.

```
.
├── assets                # Images for README.md
├── LICENSE
├── README.md
├── CITATION.cff
├── .gitignore
├── .github

# below is for the PyPI package marlin-pytorch
├── src                   # Source code for marlin-pytorch
├── tests                 # Unittest
├── requirements.lib.txt
├── setup.py
├── init.py
├── version.txt

# below is for the paper implementation
├── configs              # Configs for experiments settings
├── model                # Marlin models
├── preprocess           # Preprocessing scripts
├── dataset              # Dataloaders
├── utils                # Utility functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script

```

## Use `marlin-pytorch` for Feature Extraction

Requirements:
- Python >= 3.6, < 3.11
- PyTorch >= 1.8
- ffmpeg


Install from PyPI:
```bash
pip install marlin-pytorch
```

Load MARLIN model from online
```python
from marlin_pytorch import Marlin
# Load MARLIN model from GitHub Release
model = Marlin.from_online()
```

Load MARLIN model from file
```python
from marlin_pytorch import Marlin
# Load MARLIN model from local file
model = Marlin.from_file("path/to/marlin.pt")
# Load MARLIN model from the ckpt file trained by the scripts in this repo
model = Marlin.from_ckpt("path/to/marlin.ckpt")
```

When MARLIN model is retrieved from GitHub Release, it will be cached in `.marlin`. You can remove marlin cache by
```python
from marlin_pytorch import Marlin
Marlin.clean_cache()
```

Extract features from cropped video file
```python
# Extract features from facial cropped video with size (224x224)
features = model.extract_video("path/to/video.mp4")
print(features.shape)  # torch.Size([T, 768])
```

Extract features from in-the-wild video file
```python
# Extract features from in-the-wild video with various size
features = model.extract_video("path/to/video.mp4", crop_face=True)
print(features.shape)  # torch.Size([T, 768])
```

Extract features from video clip tensor
```python
# Extract features from clip tensor with size (B, 3, 16, 224, 224)
x = ...  # video clip
features = model.extract_features(x)  # torch.Size([B, 1568, 768])
features = model.extract_features(x, keep_seq=False)  # torch.Size([B, 768])
```

## Paper Implementation

### Requirements
- Python >= 3.7, < 3.11
- PyTorch ~= 1.11
- Torchvision ~= 0.12

### Installation

Firstly, make sure you have installed PyTorch and Torchvision with or without CUDA. 

Clone the repo and install the requirements:
```bash
git clone https://github.com/ControlNet/MARLIN.git
cd MARLIN
pip install -r requirements.txt
```

### MARLIN Pretraining

Download the [YoutubeFaces](https://www.cs.tau.ac.il/~wolf/ytfaces/) dataset (only `frame_images_DB` is required). 

Download the face parsing model from [face_parsing.farl.lapa](https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt)
and put it in `utils/face_sdk/models/face_parsing/face_parsing_1.0`.

Then run scripts to process the dataset:
```bash
python preprocess/ytf_preprocess.py --data_dir /path/to/youtube_faces
```
After processing, the directory structure should be like this:
```
├── YoutubeFaces
│   ├── frame_images_DB
│   │   ├── Aaron_Eckhart
│   │   │   ├── 0
│   │   │   │   ├── 0.555.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── crop_images_DB
│   │   ├── Aaron_Eckhart
│   │   │   ├── 0
│   │   │   │   ├── 0.555.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── face_parsing_images_DB
│   │   ├── Aaron_Eckhart
│   │   │   ├── 0
│   │   │   │   ├── 0.555.npy
```


## References
If you find this work useful in your research, please cite it.
```bibtex
@article{cai2022marlin,
  title = {MARLIN: Masked Autoencoder for facial video Representation LearnINg},
  author = {Cai, Zhixi and Ghosh, Shreya and Stefanov, Kalin and Dhall, Abhinav and Cai, Jianfei and Rezatofighi, Hamid and Haffari, Reza and Hayat, Munawar},
  journal = {arXiv preprint arXiv:2211.06627},
  year = {2022},
}
```

## Acknowledgements

Some code about model is based on [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE). The code related to preprocessing
is borrowed from [JDAI-CV/FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo).
