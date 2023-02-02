import datetime
import yaml
import argparse

from trainer import ExpMultiGpuTrainer

import os


def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="/home/ywang/RECCE/config/Recce.yml",
                        help="Specified the path of configuration file to be used.")
    parser.add_argument("--local_rank", default=0,
                        type=int,
                        help="Specified the node rank for distributed training.")
    return parser.parse_args()


if __name__ == '__main__':
    import torch

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=36000), world_size=world_size, rank=rank) #481320
        torch.distributed.barrier()
        device = torch.device(int(os.environ['LOCAL_RANK']))
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config["config"]["local_rank"] = arg.local_rank 
    config["config"]["rank"] = rank
    config["config"]["world_size"] = world_size

    trainer = ExpMultiGpuTrainer(config, stage="Train") ##
    trainer.train()

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 train.py --config /home/ywang/RECCE/config/Recce.yml
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 12345 train.py --config /home/ywang/RECCE/config/Recce.yml