import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="small", choices=["small", "base", "large"])
args = parser.parse_args()

match args.model:
    case "small":
        from marlin_configs.vit_small import vit_small_config as config
    case "base":
        from marlin_configs.vit_base import vit_base_config as config
    case "large":
        from marlin_configs.vit_large import vit_large_config as config
from marlin_huggingface import MarlinModel

WEIGHT_FILE = f"./marlin_vit_{args.model}_ytf.encoder.pt"

model = MarlinModel(config)
state_dict = torch.load(WEIGHT_FILE, map_location='cpu')
model.marlin.load_state_dict(state_dict)

model.save_pretrained(
    f"marlin_vit_{args.model}_ytf",
    config=config,
    safe_serialization=True,
)
model.push_to_hub(f"marlin_vit_{args.model}_ytf")
