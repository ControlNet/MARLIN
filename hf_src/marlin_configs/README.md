# Configurations for HuggingFace Publish

Reference [here](src/marlin_pytorch/config.py).

Publish example for vit-small, before publish, pretrained model is needed. (could be found at release)

For huggingface-hub login, documentation at [here](https://huggingface.co/docs/transformers/v4.50.0/en/custom_models?resnet=ResnetModel&push=notebook#upload)

Publishation example for `marlin_vit_small_ytf.encoder`:

```python
import torch

from vit_small import vit_small_config
from marlin_huggingface import MarlinModel, MarlinConfig

WEIGHT_FILE = "./marlin_vit_small_ytf.encoder.pt"

model = MarlinModel(vit_small_config)
state_dict = torch.load(WEIGHT_FILE, map_location='cpu')
model.marlin.load_state_dict(state_dict)

model.save_pretrained(
    "marlin_vit_small_ytf_encoder",
    config=vit_small_config,
    safe_serialization=True,
)
model.push_to_hub("marlin_vit_small_ytf_encoder")
```

Evaluate example:

```python
import torch

from transformers import AutoModel

marlin_model = AutoModel.from_pretrained(
    "<publisher>/marlin_vit_small_ytf_encoder",
)
tensor = torch.rand([1, 3, 16, 224, 224])
output = marlin_model.extract_features(tensor)

print(output.shape)
```