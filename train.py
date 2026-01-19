# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

import os
import warnings

warnings.filterwarnings("ignore")
import yaml
import time, torch, sys
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
import numpy as np
from data.dataset_e2e import get_dataloaders
import data.dataset_e2e as dp
from sklearn.metrics import roc_curve, auc
from models.model import MVT
from models.config import CfgNode
import json, glob
import datetime
import wandb # Keep import for type hinting, but make usage conditional

def get_filename_datetime():

    now = datetime.datetime.now()

    filename = now.strftime("%Y-%m-%d_%H-%M-%S")  # Example format: YYYY-MM-DD_HH-MM-SS
    
    return filename

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True

def build_network(args, config, mode, local_rank):

    model = MVT(config, mode)
    model = model.to(args.device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],find_unused_parameters=True
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model # For single-GPU, model_without_ddp is just the model itself

    if args.resume_ckpt is not None:

        ckpt = torch.load(args.resume_ckpt, map_location=lambda storage, loc: storage)

        if args.distributed:
            model.module.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt["model"])

        if is_main_process():  print (f'model loaded successfully from {args.resume_ckpt}')

    return model

    
def evaluate_score(config, test_dataset, model, use_wandb = True):

    # model.eval()
    
    gt_labels_list = []
    pred_labels_list = []
    pred_score_list = []

    scene_paths = []
    
    for batch in tqdm(test_dataset):

        with torch.no_grad():
            
            try:
                rgb_images = batch['rgb_images'].cuda()
                targets = batch['targets']
                cam_RT = batch['cam_RT']
                cam_K = batch['cam_K']
                ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()

                output, _ = model.eval_cls(rgb_images, ref_cam_P, targets, threshold = config.model.threshold, sparsity_constant = config.model.sparsity_constant, visualize = False)

                gt_labels_list.extend(output['gt_labels'].split(output['num_objects']))
                pred_labels_list.extend(output['pred_labels'].split(output['num_objects']))
                pred_score_list.extend(output['pred_logits'].split(output['num_objects']))
                scene_paths.extend(batch['scene_path'])

            except Exception as e:
                print (f'bad data encountered in evaluation loop: {e}')


    gt_labels_list1 = torch.concat(gt_labels_list)
    pred_labels_list1 = torch.concat(pred_labels_list)
    pred_score_list1 = torch.concat(pred_score_list)

    total_acc = ((gt_labels_list1==pred_labels_list1)).float().mean()

    fpr, tpr, _ = roc_curve(gt_labels_list1.cpu(), pred_score_list1.cpu()) 
    roc_auc = auc(fpr, tpr)
    
    metric_dict = {'auroc': roc_auc, 'accuracy':total_acc}

    print (metric_dict)

    if is_main_process():
        print("\n--- Evaluation Metrics ---")
        for key, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"{key.replace('_', ' ').title()}: {value.item():.4f}")
                if use_wandb:
                    wandb.log({f"eval/{key}": value.item()}, commit=False)
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
                if use_wandb:
                    wandb.log({f"eval/{key}": value}, commit=False)
        print("------------------------\n")
        if use_wandb:
            wandb.log({}, commit=True)

    return metric_dict


def generate_3d_boxes_full_dataset(config, dataset, model):
    ious_list = []
    card_acc_list = []
    output_bbox_dict = {}
    
    for batch in tqdm(dataset):

        with torch.no_grad():
            
            try:
                rgb_images = batch['rgb_images'].cuda()
                target_mask = batch['batch_silhouettes'].cuda()
                targets = batch['targets']
                cam_RT = batch['cam_RT']
                cam_K = batch['cam_K']
                ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()

                output, metric, _ = model.eval_3d(rgb_images, ref_cam_P, cam_RT, cam_K, targets, \
                    threshold = config.model.threshold, sparsity_constant = config.model.sparsity_constant, estimate_number_objs = False, visualize = False)

                assert len(output['pred_boxes']) == len(batch['targets']) 

                for path_i, pred_boxes_i, card_i, iou_i in zip(batch['scene_path'], output['pred_boxes'], metric['cardinality'], metric['IoU']):

                    box_ = pred_boxes_i['boxes'].view(-1, 2, 3).detach().cpu().numpy()
                    
                    output_bbox_dict[path_i] = {'boxes': box_.tolist(), 'IoU':iou_i, 'cardinality': card_i}

                ious_list.extend(metric['IoU'])
                card_acc_list.extend(metric['cardinality'])
   
            except Exception as e:
                print (f'bad data encountered in eval loop: {e}')

            
    miou = np.mean(ious_list)
    cardinality_accuracy = np.mean(card_acc_list)
    
    metric_dict = {'miou' : miou, 'cardinality_accuracy':cardinality_accuracy}
    print (metric_dict)

    return output_bbox_dict

def train(config, train_dataset, test_dataset, model, training_mode, device):

    i = 0
    test_loader = iter(test_dataset)

    for epoch in range(args.epochs):

        if is_main_process(): print ('#Epoch - '+str(epoch))
        model.train()

        start_time = time.time()

        #train_dataset.sampler.set_epoch(epoch)     

        for batch in train_dataset:
            
            try:
                i = i + 1
                rgb_images = batch['rgb_images'].cuda()
                target_mask = batch['batch_silhouettes'].cuda()
                targets = batch['targets']
                cam_RT = batch['cam_RT']
                cam_K = batch['cam_K']
                ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()

                loss_dict, log_dict = \
                    model.forward_alpha_ntx(rgb_images, ref_cam_P, targets, target_mask, cam_RT, cam_K, current_step=epoch)

            except Exception as e:
                print (f'bad data encountered in training loop: {e}')


            if i % 100 == 0 and is_main_process():
                
                try:
                    try:
                        test_batch = next(test_loader)
                    except StopIteration:
                        test_loader = iter(test_dataset)  # restart
                        test_batch = next(test_loader)
            
                    rgb_images = test_batch['rgb_images'].cuda()
                    target_mask = test_batch['batch_silhouettes'].cuda()
                    targets = test_batch['targets']
                    cam_RT = torch.Tensor(test_batch['cam_RT']).cuda()
                    cam_K = torch.Tensor(test_batch['cam_K']).cuda()
                    ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()

                    if training_mode == '3d_training':

                        _, metric, _ = model.eval_3d(rgb_images, ref_cam_P, cam_RT, cam_K, targets, threshold =  config.model.threshold, sparsity_constant = config.model.sparsity_constant, visualize = False)
                        log_dict['batch_cardinality_accuracy_test'] = np.mean(metric['cardinality'])
                        log_dict['batch_3d_box_IoU_test'] = np.mean(metric['IoU'])

                    if training_mode == 'cls_training':

                        output, _ = model.eval_cls(rgb_images, ref_cam_P, targets, threshold =  config.model.threshold, sparsity_constant = config.model.sparsity_constant, visualize = False)
                        fpr, tpr, _ = roc_curve(output['gt_labels'].cpu(), output['pred_logits'].cpu())
                        roc_auc = auc(fpr, tpr)
                        accuracy = ((output['gt_labels'].cpu()==output['pred_labels'].cpu())).float().mean()
                        log_dict['batch_auroc_test'] = roc_auc
                        log_dict['batch_accuracy_test'] = accuracy

                    torch.cuda.empty_cache() # Clear CUDA cache

                except Exception as e:
                    print (e)
                
                log_info = {i:np.round(j.item(),4) if isinstance(j, torch.Tensor) else np.round(j,4) for i,j in ({**loss_dict, **log_dict}).items()}
                print (f'[Epoch:{epoch}] [Step:{i}] [Time:{np.round(time.time() - start_time, 1)}s] logged info: {log_info}')
                
                if is_main_process() and args.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in log_info.items()}, commit=True) # Commit training metrics
                
                start_time = time.time()

        if training_mode == 'cls_training' and is_main_process():
            
            evaluate_score(config, test_dataset, model, use_wandb=args.use_wandb)

        torch.cuda.empty_cache() # Clear CUDA cache after evaluation

        if (epoch)%5 == 0 and is_main_process():
        
            model_module = model
            torch.save(
                {
                    "model": model_module.state_dict(),
                },
                args.ckpt_path + f"/model_{str(epoch).zfill(6)}.pt"

            )

    model_module = model
    torch.save(
        {
            "model": model_module.state_dict(),
        },
        args.ckpt_path + f"/model_{str(epoch).zfill(6)}.pt"

    )

def main(args, config, training_mode):

    if args.distributed: 
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0

    model_loaded = build_network(args, config, training_mode, local_rank)
    model = model_loaded.module if args.distributed else model_loaded

    if training_mode == 'cls_training':

        if args.resume_ckpt is None:
            raise ValueError("A resume_ckpt must be provided for 'cls_training' mode. This file comes from the '3d_training'.")

        bbox_json_file = os.path.join(
            args.exp_path,
            f"saved_3d_bbox.json"
        )

        if not os.path.isfile(bbox_json_file):

            scene_paths = [i for i in glob.glob(config.dataset.data_path + '/*') if os.path.isfile(i + '/scene3d.metadata.json')]

            _dataset = dp.dataloader(scene_paths, config, training_mode, None, is_train = False)
            data_loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size = config.dataset.test_batch_size,
                sampler=dp.data_sampler(_dataset, shuffle=True, distributed=False),
                collate_fn=dp.collate_fn,
                drop_last=False,
                pin_memory = False,
                num_workers=config.dataset.num_workers,
            )    
            
            if is_main_process(): print("Generating 3D bounding boxes for the full dataset. This may take some time...")
            bbox_json_dict = generate_3d_boxes_full_dataset(config, data_loader, model) ## save predicted 3d boxes beforehand to reuse during training

            with open(bbox_json_file, "w") as f:
                json.dump(bbox_json_dict, f, indent=4)

            del bbox_json_dict
            if is_main_process(): print (f'Created file {bbox_json_file}')
        else:
            if is_main_process(): print (f'Reusing saved 3D bounding boxes from {bbox_json_file}')
                
    elif training_mode == '3d_training':
        bbox_json_file = None 

    else:
        raise ValueError(f"Invalid mode for training: {training_mode}. Expected 'cls_training' and '3d_training'")

    dataloader_dict = get_dataloaders(config, mode = training_mode, bbox_json_file = bbox_json_file, is_distributed = args.distributed)

    train_loaders = [value for key,value in dataloader_dict.items() if 'train' in key]
    test_loaders = [value for key,value in dataloader_dict.items() if 'test' in key]

    train(
        config, train_loaders[0], test_loaders[0], model, training_mode, args.device
    )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='OddOneOut')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--mode', type=str, default='cls_training')
    parser.add_argument('--config_path', type=str, default='configs/default_toys8k.yaml')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging.')

    args = parser.parse_args()

    print ('Experiment: '+ args.exp_name)

    with open(args.config_path) as file:
        config = CfgNode(yaml.safe_load(file))
    
    if args.distributed:  init_distributed()

    cur_time = get_filename_datetime()

    args.exp_path = f'experiments/{args.exp_name}-{args.mode}'
    args.ckpt_path = f'experiments/{args.exp_name}-{args.mode}/run_{cur_time}/checkpoints'

    if is_main_process():
        print(f"Checkpoints will be saved to: {args.ckpt_path}")
        os.makedirs(args.ckpt_path, exist_ok = True)
        
        if args.use_wandb:
            if os.environ.get("WANDB_API_KEY"):
                wandb.login(key=os.environ.get("WANDB_API_KEY"))
                wandb.init(project=args.exp_name, config=vars(args))
                wandb.run.log_code(".")
            else:
                args.use_wandb = False
                print("WANDB_API_KEY environment variable not found. Disabling Weights & Biases logging. Please set WANDB_API_KEY to enable it.")

    main(args, config, args.mode)

    if is_main_process() and args.use_wandb:
        wandb.finish()
