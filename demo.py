# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

import os
import torch
import yaml
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
import glob
import cv2
from models.config import CfgNode
from models.model import MVT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ODD-ONE-OUT Inference Demo')
    parser.add_argument('--scene_path', type=str,
                        help='Path to the scene directory (e.g., data/toys8k_data/scene_00000)')
    parser.add_argument('--config_path', type=str, default='configs/default_toys8k.yaml',
                        help='Path to the model configuration YAML file')
    parser.add_argument('--resume_ckpt', type=str,
                        help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference (cuda or cpu)')

    args = parser.parse_args()

    scene_path = args.scene_path

    # Load configuration
    with open(args.config_path) as file:
        config = CfgNode(yaml.safe_load(file))

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build model
    model = MVT(config, mode='cls_training')
    model = model.to(device)

    # Load checkpoint
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Model loaded successfully from {args.resume_ckpt}")
    else:
        print("No checkpoint provided. Model will use random weights.")
        exit() # Exit if no checkpoint is provided for inference

    model.eval() # Set model to evaluation mode


    num_views = 5
    dimension = config.dataset.dimension

    print(f"Loading data for scene: {scene_path}")

    rgb_images_paths = sorted(glob.glob(os.path.join(scene_path, 'RGB', '*')))
    seg_images_paths = sorted(glob.glob(os.path.join(scene_path, 'masks', '*')))

    with open(os.path.join(scene_path, 'scene3d.metadata.json'), 'r') as f:
        json_dict = json.load(f)

    RT_matrices = [np.array(cam_pose['rotation']) for cam_pose in json_dict['camera']['poses']]
    K_matrices = [np.array(json_dict['camera']['K'])] * len(RT_matrices)
    
    # Randomly select views for consistency with training/testing setup
    view_indices = np.random.choice(len(rgb_images_paths), num_views, replace=False)
    
    batch_rgb_images_path = np.array([rgb_images_paths[i] for i in view_indices])
    batch_RT = [RT_matrices[i] for i in view_indices]
    batch_K = [K_matrices[i] for i in view_indices]

    get_transform = transforms.Compose([
        transforms.Resize((dimension, dimension), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

    rgb_images_list = []
    for rgb_path in batch_rgb_images_path:
        img = Image.open(rgb_path).convert('RGB')
        rgb_images_list.append(get_transform(img))
    batch_rgb = torch.stack(rgb_images_list)

    silhouettes_list = []
    for i in view_indices:
        seg_img = Image.open(seg_images_paths[i])
        silhouettes_list.append(transforms.ToTensor()(seg_img).squeeze(0))
    batch_silhouettes = torch.stack(silhouettes_list)
    
    object_box = np.array([objs['bbox'] for objs in json_dict['objects']])
    object_label = np.array([1 if ("-" in os.path.basename(objs['path'])) else 0 for objs in json_dict['objects']])

    # Prepare data for model input (add batch dimension)
    rgb_images_input = batch_rgb.unsqueeze(0)
    cam_RT_input = torch.Tensor(batch_RT).unsqueeze(0)
    cam_K_input = torch.Tensor(batch_K).unsqueeze(0)

    _box = torch.Tensor(object_box).flatten(1,2)

    targets_dict = {'boxes': _box, 'labels': torch.Tensor(object_label).long()}
    
    ref_cam_P_input = torch.Tensor(cam_K_input @ cam_RT_input[:,:,:3]).to(device)

    targets_input = [targets_dict]

    # --- Perform inference ---
    output, metric, log_images_3d = model.eval_3d(rgb_images_input.to(device), \
                                               ref_cam_P_input.to(device), \
                                               cam_RT_input.to(device), \
                                               cam_K_input.to(device), \
                                               targets_input, \
                                               threshold = config.model.threshold, 
                                               sparsity_constant = config.model.sparsity_constant, 
                                               visualize = True, 
                                               num_visualized_views = 2)


    output, log_images_cls = model.eval_cls(rgb_images_input.to(device), \
                                               ref_cam_P_input.to(device), \
                                               targets_input, \
                                               threshold = config.model.threshold, 
                                               sparsity_constant = config.model.sparsity_constant, 
                                               visualize = True, 
                                               num_visualized_views = 2)
                                               
    output_dir = 'output_visualizations'
    os.makedirs(output_dir, exist_ok = True)
    
    log_images = {**log_images_3d, **log_images_cls}
    scene_name = os.path.basename(scene_path)

    for key, img_data in log_images.items():

        if isinstance(img_data, torch.Tensor):
            img_data = img_data.permute(1, 2, 0).cpu().numpy() # Assuming CxHxW to HxWxC
        
        if img_data.dtype == np.float32 or img_data.dtype == np.float64:
            img_data = (img_data * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV saving if the image has 3 channels
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
        filename = os.path.join(output_dir, f"{scene_name}_{key}.png")
        cv2.imwrite(filename, img_data)
        print(f"Saved {filename}")
