# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

from train import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import numpy as np
from collections import Counter
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score
from utils import misc
import numpy as np
import time


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='help')

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/default_toys8k.yaml')

    args = parser.parse_args()

    with open(args.config_path) as file:
        config = CfgNode(yaml.safe_load(file))


    if args.distributed: 
        init_distributed()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0

    mode = 'cls_training'

    dataloader_dict = get_dataloaders(config, mode = mode, bbox_json_file = None, is_distributed = args.distributed)

    model_loaded = build_network(args, config, mode, local_rank)

    model = model_loaded.module if args.distributed else model_loaded

    model.eval()

    for key in dataloader_dict:
        if 'test' in key:
            print (f'Evaluating test set : {key}')
            evaluate_score(config, dataloader_dict[key], model, use_wandb=False)

