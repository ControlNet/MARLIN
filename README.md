# MARLIN

## Requirements

- Python >= 3.6, < 3.11
- PyTorch >= 1.8
- ffmpeg

## Getting Started

Install from PyPI:
```bash
pip install marlin-pytorch
```

Load MARLIN model and extract features
```python
from marlin_pytorch import Marlin
# Load MARLIN model from GitHub Release
model = Marlin.from_online()
# Extract features from facial cropped video with size (224x224)
features = model.extract_video("path/to/video.mp4")
print(features.shape)  # torch.Size([T, 768])

# Extract features from clip tensor with size (B, 3, 16, 224, 224)
x = ...  # video clip
features = model.extract_features(x)  # torch.Size([B, 1568, 768])
features = model.extract_features(x, keep_seq=False)  # torch.Size([B, 768])
```

Load MARLIN model from file
```python
from marlin_pytorch import Marlin
# Load MARLIN model from local file
model = Marlin.from_file("path/to/marlin.pt")
```

When MARLIN model is retrieved from GitHub Release, it will be cached in `.marlin`. You can remove marlin cache by
```python
from marlin_pytorch import Marlin
Marlin.clean_cache()
```
