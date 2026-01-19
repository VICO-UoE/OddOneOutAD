# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Volumes
from pytorch3d.renderer import VolumeRenderer, NDCGridRaysampler, EmissionAbsorptionRaymarcher
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
import numpy as np
from pytorch3d.renderer.cameras import look_at_view_transform
from einops import rearrange
import math

__LOG10 = math.log(10)


def mse2psnr(x):
    return -10.*torch.log(x)/__LOG10
    

def sample_random_camera_RT(num_cameras = 10, radius = 1):

    theta = torch.FloatTensor(num_cameras).uniform_(0, 2*np.pi)
    phi = torch.FloatTensor(num_cameras).uniform_(0, np.pi)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    R, T = look_at_view_transform(
    )
    T = T.repeat(num_cameras, 1) 
    T[:, 0] = x
    T[:, 1] = y
    T[:, 2] = z
    cameras = look_at_view_transform(
        eye=T, 
    )
    cam_RT_ = torch.eye(4,4).repeat(num_cameras,1,1)
    R, T = cameras
    cam_RT_[:,:3] = torch.cat([R,T[:,:,None]], -1)

    return cam_RT_


def convert_to_pt_cam(cam_RT, cam_K, size = 128, stride = 2):
    R = torch.Tensor(cam_RT[:,:3,:3])         
    T = torch.Tensor(cam_RT[:,:3,3]) 
    K = torch.Tensor(cam_K)
    K[:,:2] = K[:,:2]/stride
    cam = cameras_from_opencv_projection(R.cuda(), T.cuda(), camera_matrix = K.cuda(), \
        image_size = torch.tensor([size]*2).unsqueeze(0).repeat(R.shape[0],1).cuda())

    return cam



class VolRenderFeatLoss(nn.Module):
    def __init__(self, config):
        super(VolRenderFeatLoss, self).__init__()

        self.img_size = config.model.render.out_feat_Size
        self.render_dimension = self.img_size
        self.single_voxel_size = config.model.backbone.voxel_size
        self.voxel_dim = max(config.model.backbone.voxel_dim)
        self.out_feat_dim = config.model.backbone.voxel_proj_dim
        self.raySampler = NDCGridRaysampler(image_width=self.render_dimension,
                                            image_height=self.render_dimension,
                                            n_pts_per_ray=config.model.render.n_pts_per_ray,
                                            min_depth=0.0,
                                            max_depth=self.voxel_dim*self.single_voxel_size)
        self.rayMarcher = EmissionAbsorptionRaymarcher()
        self.renderer = VolumeRenderer(raysampler=self.raySampler, raymarcher=self.rayMarcher)
        self.dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', config.model.render.dinov2_model_name)
        self.stride = config.dataset.dimension//self.render_dimension

    def forward(self, radiance_volume, target_images = None, target_mask = None, cam_RT = None, cam_K = None):

        feature_3d, density_3d = radiance_volume
        N = cam_RT.shape[1] 
        B, _, D, H, W = density_3d.shape

        pt_cams = convert_to_pt_cam(cam_RT.reshape(B*N, 4, 4), cam_K.reshape(B*N, 3, 3), size = self.render_dimension, stride = self.stride)
        density_stacked = density_3d.unsqueeze(1).repeat(1,N,1,1,1,1).reshape(B*N,1, D, H, W)
        features_stacked = feature_3d.unsqueeze(1).repeat(1,N,1,1,1,1).reshape(B*N,self.out_feat_dim, D, H, W)
        volume_pt = Volumes(densities=density_stacked, features = features_stacked,  voxel_size=self.single_voxel_size)
        rendered_images, rendered_silhouettes = self.renderer(cameras=pt_cams.cuda(), volumes=volume_pt)[0].split([self.out_feat_dim, 1], dim=-1)        
        rendered_silhouettes = rendered_silhouettes.permute(0,3,1,2).contiguous()
        rendered_silhouettes =  F.upsample(rendered_silhouettes, size=[self.img_size]*2, mode='bilinear')
        rendered_images = rendered_images.permute(0,3,1,2).contiguous()
        rendered_images =  F.upsample(rendered_images, size=[self.img_size]*2, mode='bilinear')

        if target_images!=None and target_mask!=None:
            target_mask = target_mask.reshape(B*N, 1, *target_mask.shape[-2:])
            target_images = target_images.reshape(B*N, 3, *target_images.shape[-2:])

            with torch.no_grad():
                target_images_resized = F.interpolate(target_images, [self.img_size*14]*2, mode='bilinear')
                features_dict = self.dinov2_vitg14.forward_features(target_images_resized)
                target_features = features_dict['x_norm_patchtokens'].detach()
                target_features = rearrange(target_features, 'b (fh fw) d -> b d fh fw', \
                    fh = int(math.sqrt(target_features.shape[1])))

                target_images_resized = F.interpolate(target_images, [self.img_size]*2, mode='bilinear')

            loss_feat = (1-F.cosine_similarity(rendered_images, target_features, 1)).mean() 
            psnr = mse2psnr(loss_feat)

            return loss_feat, psnr, [rendered_images.detach(), target_features.detach()]

        else:

            return rendered_images.detach()


class VolRenderRGBLoss(nn.Module):
    def __init__(self, config):
        super(VolRenderRGBLoss, self).__init__()

        self.img_size = config.model.render.out_img_size
        self.render_dimension = self.img_size//2
        self.single_voxel_size = config.model.backbone.voxel_size
        self.voxel_dim = max(config.model.backbone.voxel_dim)
        self.out_feat_dim = 3
        self.raySampler = NDCGridRaysampler(image_width=self.render_dimension,
                                            image_height=self.render_dimension,
                                            n_pts_per_ray=config.model.render.n_pts_per_ray,
                                            min_depth=0.0,
                                            max_depth=self.voxel_dim*self.single_voxel_size)
        self.rayMarcher = EmissionAbsorptionRaymarcher()
        self.renderer = VolumeRenderer(raysampler=self.raySampler, raymarcher=self.rayMarcher)
        self.stride = config.dataset.dimension//self.render_dimension

    def forward(self, radiance_volume, target_images = None, target_mask = None, cam_RT = None, cam_K = None):

        feature_3d, density_3d = radiance_volume
        N = cam_RT.shape[1] 
        B, _, D, H, W = density_3d.shape

        pt_cams = convert_to_pt_cam(cam_RT.reshape(B*N, 4, 4), cam_K.reshape(B*N, 3, 3), size = self.render_dimension, stride = self.stride)
        density_stacked = density_3d.unsqueeze(1).repeat(1,N,1,1,1,1).reshape(B*N,1, D, H, W)
        features_stacked = feature_3d.unsqueeze(1).repeat(1,N,1,1,1,1).reshape(B*N,self.out_feat_dim, D, H, W)
        volume_pt = Volumes(densities=density_stacked, features = features_stacked,  voxel_size=self.single_voxel_size)
        rendered_images, rendered_silhouettes = self.renderer(cameras=pt_cams.cuda(), volumes=volume_pt)[0].split([self.out_feat_dim, 1], dim=-1)
        rendered_silhouettes = rendered_silhouettes.permute(0,3,1,2).contiguous()
        rendered_silhouettes =  F.upsample(rendered_silhouettes, size=[self.img_size]*2, mode='bilinear')
        rendered_images = rendered_images.permute(0,3,1,2).contiguous()
        rendered_images =  F.upsample(rendered_images, size=[self.img_size]*2, mode='bilinear')

        if target_images!=None and target_mask!=None:
            target_mask = target_mask.reshape(B*N, 1, *target_mask.shape[-2:])
            target_images = target_images.reshape(B*N, 3, *target_images.shape[-2:])

            with torch.no_grad():
                target_mask_resized = F.interpolate(target_mask, [self.img_size]*2, mode='nearest')
                target_images_resized = F.interpolate(target_images, [self.img_size]*2, mode='bilinear')*target_mask_resized

            loss_rgb = F.mse_loss(rendered_images, target_images_resized)
            loss_mask = F.mse_loss(rendered_silhouettes, target_mask_resized)
            psnr = mse2psnr(loss_rgb)

            return loss_rgb + loss_mask, psnr, [rendered_images.detach(), target_images_resized.detach()]

        else:

            return rendered_images.detach()


class VolRenderSeg(nn.Module):
    def __init__(self, config):
        super(VolRenderSeg, self).__init__()

        self.img_size = config.model.render.out_img_size
        self.render_dimension = self.img_size // 2
        self.single_voxel_size = config.model.backbone.voxel_size
        self.voxel_dim = max(config.model.backbone.voxel_dim)
        self.raySampler = NDCGridRaysampler(
            image_width=self.render_dimension,
            image_height=self.render_dimension,
            n_pts_per_ray=config.model.render.n_pts_per_ray,
            min_depth=0.0,
            max_depth=self.voxel_dim * self.single_voxel_size
        )
        self.rayMarcher = EmissionAbsorptionRaymarcher()
        self.renderer = VolumeRenderer(raysampler=self.raySampler, raymarcher=self.rayMarcher)
        self.stride = config.dataset.dimension // self.render_dimension

    def forward(self, mask_volume, cam_RT, cam_K, target_mask=None):
        """
        mask_volume: [B, D, H, W] segmentation mask (integer labels)
        cam_RT: [B, N, 4, 4]
        cam_K: [B, N, 3, 3]
        target_mask: optional [B, N, H, W] for loss computation
        """
        B, D, H, W = mask_volume.shape
        N = cam_RT.shape[1]
        device = mask_volume.device

        # auto-detect number of classes
        num_classes = int(mask_volume.max().item()) + 1
        feature_3d = F.one_hot(mask_volume.long(), num_classes) 
        feature_3d = feature_3d.permute(0, 4, 1, 2, 3).float() 
        density_3d = (mask_volume > 0).float().unsqueeze(1)

        pt_cams = convert_to_pt_cam(
            cam_RT.reshape(B*N, 4, 4),
            cam_K.reshape(B*N, 3, 3),
            size=self.render_dimension,
            stride=self.stride
        )

        density_stacked = density_3d.unsqueeze(1).repeat(1, N, 1, 1, 1, 1).reshape(B*N, 1, D, H, W)
        features_stacked = feature_3d.unsqueeze(1).repeat(1, N, 1, 1, 1, 1).reshape(B*N, num_classes, D, H, W)
        volume_pt = Volumes(densities=density_stacked, features=features_stacked, voxel_size=self.single_voxel_size)
        rendered_probs, rendered_silhouettes = self.renderer(cameras=pt_cams.cuda(), volumes=volume_pt)[0].split([num_classes, 1], dim=-1)

        rendered_probs = rendered_probs.permute(0, 3, 1, 2).contiguous()
        rendered_probs = F.interpolate(rendered_probs, size=[self.img_size]*2, mode='bilinear')

        rendered_silhouettes = rendered_silhouettes.permute(0, 3, 1, 2).contiguous()
        rendered_silhouettes = F.interpolate(rendered_silhouettes, size=[self.img_size]*2, mode='bilinear')

        if target_mask is not None:
            target_mask = target_mask.reshape(B*N, 1, *target_mask.shape[-2:])
            with torch.no_grad():
                target_mask_resized = F.interpolate(target_mask.float(), [self.img_size]*2, mode='nearest')
                target_onehot = F.one_hot(target_mask_resized.long().squeeze(1), num_classes).permute(0,3,1,2).float()

            loss = F.cross_entropy(rendered_probs, target_mask_resized.long().squeeze(1))
            return loss, [rendered_probs.detach(), target_onehot.detach()]

        else:
            return rendered_probs.detach()
