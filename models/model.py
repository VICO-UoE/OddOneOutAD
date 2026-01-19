# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat 
from models.encoder import load_encoder
from models.decoder import EncoderDecoder

from models.layers import SRTLinear
from models.render import VolRenderFeatLoss, VolRenderRGBLoss, VolRenderSeg,  sample_random_camera_RT
from utils import pca, misc
from models.xformer import SpatialTransformer
import torch
import torch.nn.functional as F

def pad_to_cube(x: torch.Tensor) -> torch.Tensor:
    """
    Zero-pad a tensor of shape (c, h, w, d) to (c, m, m, m),
    where m = max(h, w, d). Padding is symmetric, so the original
    volume is centered.
    """
    assert x.ndim == 4, "Input must be (c, h, w, d)"
    _, h, w, d = x.shape
    m = max(h, w, d)

    pad_h_total = m - h
    pad_w_total = m - w
    pad_d_total = m - d

    pad_h = (pad_h_total // 2, pad_h_total - pad_h_total // 2)
    pad_w = (pad_w_total // 2, pad_w_total - pad_w_total // 2)
    pad_d = (pad_d_total // 2, pad_d_total - pad_d_total // 2)

    padded = F.pad(x, (pad_d[0], pad_d[1],
                       pad_w[0], pad_w[1],
                       pad_h[0], pad_h[1]))
    return padded

def mask_diagonal(matrix):
    masked_matrix = [row[:] for row in matrix] 
    for i in range(len(masked_matrix)):
        masked_matrix[i][i] = -1 
    return torch.stack(masked_matrix)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RenderMLP(nn.Module):
    def __init__(self, input_dim=1536+180, hidden_dim=1536, out_ch = 3):
        super().__init__()
        self.net = nn.Sequential(
            SRTLinear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, out_ch),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx,ny,nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        projection: bx4x3 projection matrices (intrinsics@extrinsics)
        features: bxcxhxw  2d feature tensor to be backprojected into 3d

    Returns:
        volume: b x c x nx x ny x nz 3d feature volume
        valid:  b x 1 x nx x ny x nz volume.
                Each voxel contains a 1 if it projects to a pixel
                and 0 otherwise (not in view frustrum of the camera)
    """

    batch = features.size(0)
    channels = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    origin = torch.Tensor([[-nx*voxel_size/2, -ny*voxel_size/2, -nz*voxel_size/2]])
    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(batch,-1,-1) # bx3xhwd
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1)

    camera = torch.bmm(projection, world)
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    # put features in volume
    volume = torch.zeros(batch, channels, nx*ny*nz, dtype=features.dtype,  device=device)
    
    for b in range(batch):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(batch, channels, nx, ny, nz)
    valid = valid.view(batch, 1, nx, ny, nz)

    return volume, valid

def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


class MV2dto3d(nn.Module):

    def __init__(self, config):

        super(MV2dto3d, self).__init__()

        self.feat_encoder, self.backbone2d_stride = load_encoder(config.model.backbone.voxel_feat_dim)

        self.decoder = EncoderDecoder(channels=[config.model.backbone.voxel_feat_dim, 64, 128, 256], layers_down=[1, 2, 3, 4], \
                            layers_up=[3, 2, 1], norm='nnSyncBN', drop=0, zero_init_residual=True, \
                            cond_proj=True)

        self.voxel_dim_train = config.model.backbone.voxel_dim
        self.voxel_size = config.model.backbone.voxel_size
        self.origin = torch.tensor([0,0,0]).view(1,3)


        self.features_head = RenderMLP(config.model.backbone.voxel_feat_dim, config.model.backbone.voxel_hidden_dim, config.model.backbone.voxel_proj_dim)
        self.density_head = RenderMLP(config.model.backbone.voxel_feat_dim, config.model.backbone.voxel_hidden_dim, 1)
        self.rgb_head = RenderMLP(config.model.backbone.voxel_feat_dim, config.model.backbone.voxel_hidden_dim, 3)


    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        self.volume = 0
        self.valid = 0


    def inference1(self, projection, feature):
        """ Backprojects image features into 3D and accumulates them.

        This is the first half of the network which is run on every frame.
        Only pass one of image or feature. If image is passed 2D features
        are extracted from the image using self.backbone2d. When features
        are extracted external to this function pass features (used when 
        passing multiple frames through the backbone2d simultaniously
        to share BatchNorm stats).

        Args:
            projection: bx3x4 projection matrix
            image: bx3xhxw RGB image
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """

        projection = projection.clone()
        projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride

        voxel_dim = self.voxel_dim_train

        volume, valid = backproject(voxel_dim, self.voxel_size, self.origin,
                                    projection, feature)


        self.volume = self.volume + volume
        self.valid = self.valid + valid

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def forward(self, rgb_images, projection):
        
        self.initialize_volume()

        bs, num_views = rgb_images.shape[:2]
        stacked_rgb = rearrange(rgb_images, 'b n c fh fw -> (b n) c fh fw')
        img_bgr = stacked_rgb[:, [2, 1, 0], :, :]

        img_bgr = img_bgr * 255.0
        PIXEL_MEAN = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1).to(img_bgr.device)
        PIXEL_STD  = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1).to(img_bgr.device)
        img_bgr = (img_bgr - PIXEL_MEAN) / PIXEL_STD

        cnn_feats = self.feat_encoder(img_bgr)
        feats =  rearrange(cnn_feats, '(b n) c fh fw -> b n c fh fw', b = bs)

        for view_idx in range(num_views):
            self.inference1(projection[:,view_idx], feats[:,view_idx])

        volume = self.volume/self.valid
        volume = volume.transpose(0,1)
        volume[:,self.valid.squeeze(1)==0]=0
        volume = volume.transpose(0,1)

        _, _, vol_feats2 = self.decoder(volume.permute(0,1,4,3,2))

        density_3d = self.density_head(vol_feats2.permute(0,2,3,4,1)).permute(0,4,1,2,3)# .clamp(-self.CLAMP_OUTPUT_VALUE,self.CLAMP_OUTPUT_VALUE)
        feature_3d = self.features_head(vol_feats2.permute(0,2,3,4,1)).permute(0,4,1,2,3)# .clamp(-self.CLAMP_OUTPUT_VALUE,self.CLAMP_OUTPUT_VALUE)
        rgb_3d = self.rgb_head(vol_feats2.permute(0,2,3,4,1)).permute(0,4,1,2,3)# .clamp(-self.CLAMP_OUTPUT_VALUE,self.CLAMP_OUTPUT_VALUE)

        volumes = {"feat_raw_3d" : vol_feats2,
                   "feat_proj_3d" : feature_3d,
                   "rgb_3d" : rgb_3d,
                   "density_3d" : density_3d}

        return volumes


def extract_crop_objs_volumes(volume, targets, resize_dim = (8,8,8), voxel_size = 0.04, pad_len = 1):

    B, _,d_vox, h_vox, w_vox = volume.shape

    batch_list = []

    for b_id in range(B):
        objs_crops = []
        raw_boxes = targets[b_id]['boxes']
        bboxs = raw_boxes/voxel_size

        bboxs[:,:3] += torch.Tensor([[h_vox//2,w_vox//2,d_vox//2]])
        bboxs = bboxs.round().int()

        for box_i in bboxs:
            x,y,z,w,h,l,pad = *box_i,pad_len
            half_l, half_h, half_w = int(l//2+pad), int(h//2+pad), int(w//2+pad)
            vol_crop = volume[b_id,  :,  max(0,z-half_l):z+half_l+1, max(0,y-half_h):y+half_h+1,  max(0,x-half_w):x+half_w+1].clone() 
            
            vol_crop_padded = pad_to_cube(vol_crop)

            resized_crop = F.interpolate(vol_crop_padded[None], size = resize_dim, mode = 'trilinear')
            objs_crops.append(resized_crop)
    
        batch_list.append({'feat': torch.concat(objs_crops), 'boxes': raw_boxes})

    return batch_list


class MatchNet(nn.Module):

    def __init__(self, config):
        super(MatchNet, self).__init__()

        self.config = config

        n_head = config.model.matching_net.n_head
        dim_per_head = config.model.matching_net.dim_per_head
        n_layer = config.model.matching_net.n_layer

        self.d_model = config.model.backbone.voxel_feat_dim
        self.voxel_proj_dim = config.model.backbone.voxel_proj_dim

        self.n_head = n_head
        self.attn = SpatialTransformer(self.d_model, n_head, dim_per_head, n_layer, context_dim=self.d_model, use_linear = True, attn_type='softmax-xformers', use_checkpoint = False)
        self.zero_pad = nn.ConstantPad3d(config.model.matching_net.pad_len, 0)
        self.fc = MLP(self.d_model, self.d_model*8, 1, 2)

    def forward(self, input):

        _, _, d_x, d_y, d_z = input[0]['feat'].shape
        seq_length = [inp_i['feat'].shape[0] for inp_i in input]

        batch_seq_feat = [rearrange(crop_batch['feat'], 'n c h w d -> (n h w d) c') for crop_batch in input]
        batch_feat = torch.nn.utils.rnn.pad_sequence(batch_seq_feat).permute(1,0,2)

        f_q, dino_q, _ = batch_feat.split([self.d_model, self.voxel_proj_dim, 1], 2)

        topk = 5 * self.config.model.matching_net.topk ##  number_of_object * topk_per_object 

        with torch.no_grad():
            affinity = torch.einsum('bic,bjc->bij', dino_q.detach(), dino_q.detach())
            mask = -torch.inf*torch.ones_like(affinity)
            index = affinity.topk(k = topk, dim = -1, largest = True)[1]
            mask.scatter_(-1,index, 0.) 
            mask = repeat(mask, 'b n1 n2 -> (b nh) n1 n2', nh = self.n_head).detach()

        out = self.attn(f_q, f_q, mask = mask)
        feat = rearrange(out, 'b (n h w d) c -> b n c h w d', h=d_x,w=d_y,d=d_z).mean([3,4,5])

        logits = self.fc(feat)

        logits = [logit[:seq_len].squeeze(-1) for logit, seq_len in zip(logits, seq_length)]

        output = [{'logit' : logit, 'boxes':inp['boxes']} for logit, inp in zip(logits, input)]

        return output 

    def calc_loss(self, outputs, targets):

        device = outputs[0]['logit'].device

        logit_list = []
        gt_list = []

        for output, target in zip(outputs, targets):
            # Convert target and predicted boxes to corner format using the vectorized function
            tgt_boxes_corner = misc.center_to_corner(target['boxes'])
            pred_boxes_corner = misc.center_to_corner(output['boxes'])

            iou_matrix = misc.compute_iou_3d(tgt_boxes_corner, pred_boxes_corner)


            max_iou_indices = iou_matrix.argmax(dim=1)
            
            logit_list.append(output['logit'][max_iou_indices])
            gt_list.append(target['labels'].to(device))

        logits = torch.cat(logit_list)
        gt_labels = torch.cat(gt_list)

        weight = gt_labels.float().clone()
        weight[weight==1]=0.7
        weight[weight==0]=0.3

        loss = F.binary_cross_entropy_with_logits(logits, gt_labels.float(), weight = weight, reduction='none').mean()

        pred_logits = logits.sigmoid()
        pred_labels = (logits.sigmoid()>0.5).long() 

        batch_acc = (pred_labels==gt_labels).float().mean()

        lens = [len(i) for i in  gt_list]

        return loss, batch_acc, (pred_logits, pred_labels, gt_labels, lens)


class MVT(nn.Module):
    def __init__(self, config, mode = '3d_training'): # mode in ['3d_training', 'cls_training']

        super(MVT, self).__init__()

        self.mvencoder = MV2dto3d(config)
        self.matching_network = MatchNet(config)

        param_dict = {"backbone" : self.mvencoder.parameters(), "matching_net" : self.matching_network.parameters()}

        self.render_module_feature = VolRenderFeatLoss(config)
        self.render_module_rgb = VolRenderRGBLoss(config)
        self.render_module_seg = VolRenderSeg(config)
    
        params = [{"params": param_dict[arch], "lr": float(config.train[mode]['lr'][arch])} \
                for arch, _ in  config.train[mode]['lr'].items()]

        self.loss_weights = {i:j for i,j in zip(config.train[mode]['loss_types'], config.train[mode]['loss_weights'])}

        self.optim =  torch.optim.Adam(params, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.log_images = []
        self.config = config

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def forward_alpha_ntx(self, rgb_images, ref_cam_P, targets, target_mask, cam_RT, cam_K, current_step):

        if current_step is not None:
            self.current_step = current_step

        loss_dict, log_dict = {}, {}
        num_view = 5 #rgb_images.shape[1]//2

        volumes = self.mvencoder(rgb_images[:,:num_view], ref_cam_P[:,:num_view])

        [features_raw_3d, feature_3d, rgb_3d, density_3d] = \
            volumes["feat_raw_3d"], volumes["feat_proj_3d"], volumes["rgb_3d"], volumes["density_3d"]

        if "pairwise_cls_loss" in self.loss_weights.keys():

            stacked_volume = torch.concat([features_raw_3d, feature_3d, density_3d], 1)

            crops_instances = extract_crop_objs_volumes(stacked_volume, targets, \
                resize_dim = self.config.model.matching_net.crop_resize_dim, \
                    voxel_size = self.config.model.backbone.voxel_size, pad_len = 2 )

            outputs = self.matching_network(crops_instances)

            cls_loss, batch_acc, _ = self.matching_network.calc_loss(outputs, targets)

            loss_dict["cls_loss"] = self.loss_weights["pairwise_cls_loss"] * cls_loss
            log_dict["batch_acc_train"] = batch_acc

        if "render_loss" in self.loss_weights.keys():

            loss_feat, _, _ = \
                    self.render_module_feature([feature_3d, density_3d.detach()], rgb_images[:,num_view:], target_mask[:,num_view:], cam_RT[:,num_view:], cam_K[:,num_view:])

            loss_rgb, _, _ = \
                   self.render_module_rgb([rgb_3d, density_3d], rgb_images[:,num_view:], target_mask[:,num_view:], cam_RT[:,num_view:], cam_K[:,num_view:])

            loss_dict["render_loss_feat"] = self.loss_weights["render_loss"] * loss_feat
            loss_dict["render_loss_rgb"] = self.loss_weights["render_loss"] * loss_rgb

        self.loss_total = sum(loss_dict.values())

        self.optim.zero_grad()
        self.loss_total.backward()
        self.optim.step()
                
        return loss_dict, log_dict


    def eval_3d(self, rgb_images, ref_cam_P, cam_RT, cam_K, targets, threshold, sparsity_constant, estimate_number_objs = False, visualize = False, num_visualized_views = 2):

        voxel_size = self.config.model.backbone.voxel_size

        with torch.no_grad():

            volumes = self.mvencoder(rgb_images[:,:], ref_cam_P[:,:])
            self.output_volumes = volumes

            [_, feature_3d, _, density_3d] = \
                volumes["feat_raw_3d"], volumes["feat_proj_3d"], volumes["rgb_3d"], volumes["density_3d"]

            pred_boxes, labeled_mask, [ious, cardinality_acc, top_view_segmentation_masks] = misc.object_selector(density_3d, targets, visualize = visualize, \
                 thr = threshold, sparsity_constant = sparsity_constant, voxel_size = voxel_size, estimate_number_objs = estimate_number_objs)

        log_images = {}

        if visualize:
            
            log_images['obj_clustering_visualization_topview'] = misc.save_image_grid(top_view_segmentation_masks, n_cols = min(len(top_view_segmentation_masks), 4))

            rendered_image_feat = \
                    self.render_module_feature([feature_3d.detach(), density_3d.detach()], None, None, cam_RT[:,:num_visualized_views].clone(), cam_K[:,:num_visualized_views].clone())
            rendered_image_feat_reshaped = rendered_image_feat.view(rgb_images.shape[0], -1, *rendered_image_feat.shape[1:])

            rendered_seg_mask = self.render_module_seg(torch.Tensor(labeled_mask).to(rgb_images.device), cam_RT[:,:num_visualized_views].clone(), cam_K[:,:num_visualized_views].clone()).argmax(1)
            rendered_seg_mask_reshaped = rendered_seg_mask.view(rgb_images.shape[0], -1, *rendered_seg_mask.shape[1:]).cpu().detach().numpy()

            total_dinov2_viz = []
            total_segmask_viz = []
            total_rgb_inputs = []

            for index_i in range(len(rgb_images)): 

                pca_visuals = pca.pca_dense_matching_visualization(rendered_image_feat_reshaped[index_i], thres = 0)
                seg_mask = rendered_seg_mask_reshaped[index_i]
                rgb_image = rgb_images[index_i,:num_visualized_views].permute(0,2,3,1).cpu().detach().numpy()

                total_rgb_inputs.append(misc.save_image_grid([rgb_image[j] for j in range(num_visualized_views)],n_cols = min(num_visualized_views, 2)))

                overlapped_pca_visual = misc.save_image_grid([misc.overlay_images(rgb_image[j], pca_visuals[j], alpha = 0.7) for j in range(num_visualized_views)], n_cols = num_visualized_views)
                total_dinov2_viz.append(overlapped_pca_visual)
        
                seg_mask_overlapped = misc.save_image_grid([misc.visualize_segmentation(rgb_image[j], seg_mask[j], alpha = 0.7) for j in range(num_visualized_views)], n_cols = num_visualized_views)
                total_segmask_viz.append(seg_mask_overlapped)

            log_images['rgb_input'] = misc.save_image_grid(total_rgb_inputs, n_cols = min(len(total_rgb_inputs), 2))
            log_images['rendered_dinov2_feat_visualization_overlapped'] = misc.save_image_grid(total_dinov2_viz, n_cols = min(len(total_dinov2_viz), 2))
            log_images['rendered_seg_mask_overlapped'] = misc.save_image_grid(total_segmask_viz, n_cols = min(len(total_segmask_viz), 2))
            
        output = {'pred_boxes' : pred_boxes, 'labeled_3d_mask': labeled_mask}
        metric = {'IoU': [float(i) for i in ious], 'cardinality': cardinality_acc}

        return output, metric, log_images

    def eval_cls(self, rgb_images, ref_cam_P, targets, threshold, sparsity_constant, estimate_number_objs = False, visualize = False, num_visualized_views = 2):

        voxel_size = self.config.model.backbone.voxel_size

        with torch.no_grad():
            
            volumes = self.mvencoder(rgb_images[:,:], ref_cam_P[:,:])
            self.output_volumes = volumes

            [features_raw_3d, feature_3d, _, density_3d] = \
                volumes["feat_raw_3d"], volumes["feat_proj_3d"], volumes["rgb_3d"], volumes["density_3d"]

            pred_boxes, _, _ = misc.object_selector(density_3d, targets, visualize = visualize, \
                 thr = threshold, sparsity_constant = sparsity_constant, voxel_size = voxel_size, estimate_number_objs = estimate_number_objs)

            stacked_volume = torch.concat([features_raw_3d, feature_3d, density_3d], 1)
            crops_instances = extract_crop_objs_volumes(stacked_volume, pred_boxes, \
                resize_dim = self.config.model.matching_net.crop_resize_dim, \
                    voxel_size = self.config.model.backbone.voxel_size, pad_len = 2)

            outputs = self.matching_network(crops_instances)

            _, _, (pred_logits, pred_labels, gt_labels, num_objects)  = self.matching_network.calc_loss(outputs, targets)

        pred_labels_batch = pred_labels.split(num_objects)

        log_images = {}

        if visualize:
            
            bbox_visual_batch = []

            for index_i in range(len(rgb_images)): 

                pred_label = pred_labels_batch[index_i].cpu().detach().numpy()
                box_3d = targets[index_i]['boxes'].cpu().detach().numpy()
                rgb_image = rgb_images[index_i,:num_visualized_views].permute(0,2,3,1).cpu().detach().numpy()
                proj_matrix = ref_cam_P[index_i].cpu().detach().numpy()
                
                anomaly_boxes = [box_3d[j] for j in range(len(box_3d)) if pred_label[j] == 1 ]                  
                cls_box_visual = misc.save_image_grid([misc.draw_3d_bboxes(rgb_image[j], anomaly_boxes, proj_matrix[j], padding = 0.1) \
                    for j in range(num_visualized_views)], n_cols = num_visualized_views)

                bbox_visual_batch.append(cls_box_visual)

            log_images['anomaly_classification_visualization'] = misc.save_image_grid(bbox_visual_batch, n_cols = min(len(bbox_visual_batch), 2))

        output = {'pred_labels' : pred_labels, 'pred_logits': pred_logits, 'gt_labels' : gt_labels, 'num_objects' : num_objects}

        return output, log_images


    def get_current_step(self):
        return getattr(self, 'current_step', None)
