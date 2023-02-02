import yaml
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

config = "/home/ywang/RECCE/config/Recce.yml"
local_rank = 0

with open(config) as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
config["config"]["local_rank"] = local_rank

print(config)