# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

import os
import json, glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
import torch
from utils.misc import center_to_corner

def calculate_scene_bbox_and_scaling(boxes_center_format, volume_limit):
    """
    Calculates the bounding box of the entire scene and a scaling limit.

    Args:
        boxes_center_format (torch.Tensor): A tensor of bounding boxes in (cx, cy, cz, h, w, d) format.
        volume_limit (torch.Tensor): A tensor representing the maximum extent for each dimension (x, y, z).

    Returns:
        tuple: A tuple containing:
            - scene_bbox_corner (torch.Tensor): The overall scene bounding box in (x_min, y_min, z_min, x_max, y_max, z_max) format.
            - scaling_limit (float): The scaling factor needed to fit the scene within the volume limit.
    """
    # Convert all individual boxes to corner format (x_min, y_min, z_min, x_max, y_max, z_max)
    boxes_corner_format = center_to_corner(boxes_center_format)

    # Calculate the overall scene bounding box
    # Find the minimum of all x_min, y_min, z_min
    scene_x_min = torch.min(boxes_corner_format[:, 0])
    scene_y_min = torch.min(boxes_corner_format[:, 1])
    scene_z_min = torch.min(boxes_corner_format[:, 2])

    # Find the maximum of all x_max, y_max, z_max
    scene_x_max = torch.max(boxes_corner_format[:, 3])
    scene_y_max = torch.max(boxes_corner_format[:, 4])
    scene_z_max = torch.max(boxes_corner_format[:, 5])

    # Calculate the dimensions of the scene bounding box
    scene_width = scene_x_max - scene_x_min
    scene_height = scene_y_max - scene_y_min
    scene_depth = scene_z_max - scene_z_min
    scene_dimensions = torch.tensor([scene_width, scene_height, scene_depth])

    # Calculate the scaling factor for each dimension
    # Avoid division by zero for dimensions that are 0
    scaling_factors = torch.where(scene_dimensions == 0, torch.tensor(float('inf')), volume_limit / scene_dimensions)

    # The overall scaling limit is the minimum of the individual scaling factors
    scaling_limit = torch.min(scaling_factors)

    return scaling_limit

def augment_scene_RT_bbox(RT, bbox,
                          scale_range=(0.9, 1.1),
                          rot_range=(-15, 15),
                          trans_range=(-0.2, 0.2),
                          flip_prob=0.5):
    """
    Apply optional augmentations (scale, rotation, translation, flip)
    to a scene with multi-view cameras and 3D bounding boxes.
    Pass None to a range to disable that augmentation.

    Args:
        RT          : (N, 4, 4) extrinsic matrices for N cameras
        bbox        : (M, 6) boxes [cx, cy, cz, h, w, d]
        scale_range : tuple (min, max) or None
        rot_range   : tuple (min_deg, max_deg) or None
        trans_range : tuple (min, max) or None
        flip_prob   : probability of flipping along X axis, set None or 0 to disable

    Returns:
        RT_new      : (N, 4, 4) updated extrinsics
        bbox_new    : (M, 6) updated bounding boxes in [cx, cy, cz, h, w, d]
    """
    device = RT.device
    M = bbox.shape[0]

    # === 1. Random scale ===
    s = 1.0
    if scale_range is not None:
        s = torch.empty(1, device=device).uniform_(*scale_range).item()

    # === 2. Random roll rotation (around Z axis) ===
    R_z = torch.eye(3, device=device)
    if rot_range is not None:
        angle = torch.empty(1, device=device).uniform_(*rot_range) * torch.pi / 180
        cos_a = angle.cos()
        sin_a = angle.sin()
        R_z = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=torch.float32, device=device)

    # === 3. Random translation ===
    t_aug = torch.zeros(3, device=device)
    if trans_range is not None:
        t_aug = torch.empty(3, device=device).uniform_(*trans_range)
        t_aug[2] = 0 # no z axis trans

    # === 4. Optional flip along X ===
    flip = torch.eye(3, device=device)
    if flip_prob is not None and flip_prob > 0 and torch.rand(1, device=device) < flip_prob:
        flip[0, 0] = -1.0

    # === Combined world transform (rotation + flip) ===
    T_rot_flip = flip @ R_z

    # ---- Update bounding boxes (Vectorized) ----
    # Get original centers and dimensions
    centers = bbox[:, :3]  # (M, 3)
    # Reorder dims from [h, w, d] to [w, h, d] for corner calculation
    dims_whd = bbox[:, 3:][:, [1, 0, 2]] # (M, 3)

    # Create corner offsets from the origin [-1, 1]
    corner_offsets = torch.tensor([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]
    ], device=device, dtype=torch.float32) # (8, 3)

    # Calculate 8 corners for all M boxes at once. Shape: (M, 8, 3)
    corners = centers[:, None, :] + dims_whd[:, None, :] / 2 * corner_offsets[None, :, :]

    # Reshape for efficient matrix multiplication: (M * 8, 3)
    corners_flat = corners.view(-1, 3)

    # Apply transformations: rotation, flip, translation, and scale
    # Note: (A @ B.T).T is equivalent to B @ A.T
    transformed_corners = corners_flat @ T_rot_flip.T
    transformed_corners = transformed_corners + t_aug # Broadcasting t_aug
    transformed_corners = s * transformed_corners

    # Reshape back to (M, 8, 3)
    transformed_corners = transformed_corners.view(M, 8, 3)

    # Find new axis-aligned bounding box from transformed corners
    xyz_min, _ = torch.min(transformed_corners, dim=1)
    xyz_max, _ = torch.max(transformed_corners, dim=1)

    # Calculate new centers and dimensions
    new_centers = (xyz_min + xyz_max) / 2
    new_dims_whd = xyz_max - xyz_min

    # Combine into new bbox format [cx, cy, cz, h, w, d]
    # Reorder new_dims_whd [w, h, d] back to [h, w, d]
    bbox_new = torch.cat([new_centers, new_dims_whd[:, [1, 0, 2]]], dim=1)


    # ---- Update camera extrinsics ----
    # Create a single 4x4 matrix for the world augmentation
    # This matrix transforms points in the *original* world to the *new* world
    M_world_aug = torch.eye(4, device=device)
    M_world_aug[:3, :3] = T_rot_flip
    M_world_aug[:3, 3] = t_aug
    
    # The scale is applied to the world *after* the rotation and translation
    S_world_aug = torch.eye(4, device=device)
    S_world_aug[0, 0] = S_world_aug[1, 1] = S_world_aug[2, 2] = s
    
    # Combined augmentation matrix: p'_world = S_aug @ M_aug @ p_world
    M_combined_aug = S_world_aug @ M_world_aug

    # To update the camera extrinsics (world-to-camera), we must post-multiply
    # by the *inverse* of the world augmentation matrix.
    # RT_new = RT_old @ M_combined_aug_inv
    M_combined_aug_inv = torch.inverse(M_combined_aug)
    
    RT_new = RT @ M_combined_aug_inv

    return RT_new, bbox_new


def apply_random_rotation_in_space(P, XYZ, r):
    r_inv = r.inverse()
    rot_P = [P_i@r for P_i in P]
    rot_XYZ = [(r_inv@torch.cat([XYZ_i,torch.ones([1])]))[:3] for XYZ_i in XYZ]
    return torch.stack(rot_P), torch.stack(rot_XYZ)
    
def transform_points(points, transform, translate=True):
    """ Apply linear transform to a np array of points.
    Args:
        points (np array [..., 3]): Points to transform.
        transform (np array [3, 4] or [4, 4]): Linear map.
        translate (bool): If false, do not apply translation component of transform.
    Returns:
        transformed points (np array [..., 3])
    """
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = np.ones_like(points[..., :1])
    else:
        constant_term = np.zeros_like(points[..., :1])
    points = np.concatenate((points, constant_term), axis=-1)

    points = np.einsum('nm,...m->...n', transform, points)
    return points[..., :3]


def read_json_KRT(fname):
    with open(fname,'r') as f:
        data = json.load(f)

    KRT = torch.Tensor((np.array(data['K'])@np.array(data['RT'][:3])))
    return KRT

def get_transform(size=256, method=Image.BICUBIC, toTensor=True):
    transform_list = []
    
    transform_list.append(transforms.Resize((size, size), interpolation=method))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)



class dataloader(Dataset):

    def __init__(self, scene_paths, config, mode, bbox_json_file = None, is_train = True):

        path =  config.dataset.data_path
        self.is_train = is_train
        self.mode = mode
        self.config = config
        self.num_views = config.train[mode].num_views
        self.dimension = config.dataset.dimension
        self.max_objs = config.dataset.max_objs
        self.augmentation_config = config.train[mode].random_scene_augmentation
        
        self.scene_paths = scene_paths

        if bbox_json_file is not None:
            with open(bbox_json_file, "r") as f:
                self.saved_bbox = json.load(f)
                self.IoU_bar =  np.percentile([i['IoU'] for i in self.saved_bbox.values()], 2)

    def __len__(self):  
        
        return len(self.scene_paths)

    def get_image_tensor(self, path, size):
        img = Image.open(path).convert('RGB')
        trans = get_transform(size = size)
        img = trans(img)
        return img    

    def __getitem__(self, index):

        try:

            scene_path = self.scene_paths[index]

            scene_id = os.path.basename(scene_path)

            rgb_images = sorted(glob.glob(scene_path+'/RGB/*'))
            seg_images = sorted(glob.glob(scene_path+'/masks/*'))

            with open(scene_path+'/scene3d.metadata.json','r') as f:
                json_dict = json.load(f)

            RT_matrices = [np.array(cam_pose['rotation']) for cam_pose in json_dict['camera']['poses']]
            K_matrices = [np.array(json_dict['camera']['K'])]*len(RT_matrices)

            view_indices = np.random.choice(len(rgb_images), 2 * self.num_views, replace = False)  # 2 * self.num_views views cause we need self.num_views for input and rest self.num_views for GT rendering loss
            batch_rgb_images_path = np.array([rgb_images[i] for i in view_indices])
            batch_RT = [RT_matrices[i] for i in view_indices]
            batch_K = [K_matrices[i] for i in view_indices]

            batch_rgb = torch.stack([self.get_image_tensor(rgb_path, self.dimension) for rgb_path in batch_rgb_images_path])
            batch_silhouettes = torch.stack([transforms.ToTensor()(Image.open(seg_images[i]).convert("L")) for i in view_indices]).squeeze(1)
            
            if hasattr(self, "saved_bbox") and self.is_train:

                if scene_id in self.saved_bbox.keys():

                    if self.saved_bbox[scene_id]['IoU'] < self.IoU_bar:
                        return None

                    object_box = np.array(self.saved_bbox[scene_id]['boxes']) 
                else:
                    object_box = np.array([objs['bbox'] for objs in json_dict['objects']])
            else:
                object_box = np.array([objs['bbox'] for objs in json_dict['objects']])

            object_label = np.array([1 if ("-" in os.path.basename(objs['path'])) else 0 for objs in json_dict['objects']])

            _box = torch.Tensor(object_box).flatten(1,2)
            _RT = torch.Tensor(batch_RT)

            if self.is_train:

                if self.augmentation_config.scale_range == 'auto':
                    dx_max, dy_max, dz_max = self.config.model.backbone.voxel_dim
                    volume_limit =  self.config.model.backbone.voxel_size * np.array([dx_max, dy_max, dz_max//2]) # 3.84
                    max_scale = max(1.0, calculate_scene_bbox_and_scaling(_box, volume_limit = volume_limit))
                    scale_range = (1.0, max_scale)
                elif self.augmentation_config.scale_range == 'None':
                    scale_range = None
                else:
                    raise ValueError(f"Invalid scale_range configuration: {self.augmentation_config.scale_range}. Expected 'auto', 'None', or a tuple (min, max).")
                
                rot_range = tuple(map(float, self.augmentation_config.rot_range.strip("()").split(","))) if self.augmentation_config.rot_range !=  'None' else None
                trans_range = tuple(map(float, self.augmentation_config.trans_range.strip("()").split(","))) if self.augmentation_config.trans_range !=  'None' else None
                flip_prob = self.augmentation_config.flip_prob if self.augmentation_config.flip_prob !=  'None' else 0.0

                aug_RT, aug_box = augment_scene_RT_bbox(_RT, _box, scale_range = scale_range, rot_range = rot_range, trans_range = trans_range, flip_prob = flip_prob)

            else:

                aug_RT, aug_box = augment_scene_RT_bbox(_RT, _box, scale_range = None, rot_range = None, trans_range = None, flip_prob = 0.0)
                
            targets = {'boxes':aug_box, 'labels':torch.Tensor(object_label).long()}

            return_dict = {
                'rgb_images':batch_rgb,
                'cam_RT' : aug_RT,
                'cam_K' : np.array(batch_K),
                'batch_silhouettes' : batch_silhouettes,
                'targets' : targets,
                'scene_path' : scene_path
            }

            return return_dict
        except:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    batch_rgb = torch.stack([data['rgb_images'] for data in batch]) 
    batch_silhouettes = torch.stack([data['batch_silhouettes'] for data in batch]) 
    batch_RT = np.stack([data['cam_RT'] for data in batch]) 
    batch_K = np.stack([data['cam_K'] for data in batch]) 
    targets = [data['targets'] for data in batch]
    scene_path = [data['scene_path'] for data in batch]
    
    return_dict = {
            'rgb_images':batch_rgb,
            'batch_silhouettes':batch_silhouettes,
            'cam_RT' : batch_RT,
            'cam_K' : batch_K,
            'targets' : targets,
            'scene_path' : scene_path
        }

    return return_dict

def data_sampler(dataset, shuffle, distributed):

    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)

def get_dataloaders(config, mode, bbox_json_file, is_distributed):

    dataloader_dict = {}

    with open(config.dataset.split_path, "r") as f:
        split = json.load(f)

    for key in split.keys():

        scene_paths = [os.path.join(config.dataset.data_path, i) for i in split[key]]

        is_train = 'test' not in key

        _dataset = dataloader(scene_paths, config, mode, bbox_json_file, is_train = 'test' not in key)
        data_loader = torch.utils.data.DataLoader(
            _dataset,
            batch_size = config.dataset.batch_size if is_train else config.dataset.test_batch_size,
            sampler=data_sampler(_dataset, shuffle=True, distributed=is_distributed),
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory = False,
            num_workers=config.dataset.num_workers,
        )    

        dataloader_dict[key] = data_loader    

    return dataloader_dict

