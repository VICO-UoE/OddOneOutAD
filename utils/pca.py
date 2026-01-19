# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from einops import rearrange, reduce, repeat 
import numpy as np


def pca_bg_extractor_3d(target_features, thres = 10):
    """
    Extracts a foreground mask from 3D target features using PCA.
    The first principal component is used to distinguish foreground from background
    based on a threshold.

    Args:
        target_features (torch.Tensor): A 5D tensor of shape (B, C, D, H, W)
                                        representing batch, channels, depth, height, width.
        thres (int, optional): The threshold applied to the first principal component.
                               Values below this threshold are considered background.
                               Defaults to 10.

    Returns:
        torch.Tensor: A 4D boolean tensor of shape (B, D, H, W) where True indicates
                      foreground (object) and False indicates background.
    """
    B, C, D, H, W = target_features.shape

    input_tensor = target_features.detach().cpu() 
    features = rearrange(input_tensor, 'b d fd fh fw -> (b fd fh fw) d').numpy()

    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)

    pca_features_bg = pca_features[:, 0] < thres
    pca_features_fg = ~pca_features_bg

    return pca_features_fg.view(B, D, H, W)

def pca_dense_matching_visualization(target_features, thres = 10):
    """
    Generates a 2D RGB visualization by applying PCA to 2D target features.
    It first separates foreground and background based on a threshold on the
    first principal component, then re-applies PCA to the foreground features
    and scales them for RGB representation.

    Args:
        target_features (torch.Tensor): A 4D tensor of shape (B, C, H, W)
                                        representing batch, channels, height, width.
        thres (int, optional): The threshold applied to the first principal component
                               to identify background features. Defaults to 10.

    Returns:
        np.ndarray: A 4D NumPy array of shape (B, H, W, 3) representing the
                    RGB visualization of the features, scaled to 0-255.
    """
    B, _, H, W = target_features.shape

    input_tensor = target_features.detach().cpu() 
    features = rearrange(input_tensor, 'b d fh fw -> (b fh fw) d').numpy()

    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)

    pca_features_bg = pca_features[:, 0] < thres
    pca_features_fg = ~pca_features_bg

    pca.fit(features[pca_features_fg]) 
    pca_features_rem = pca.transform(features[pca_features_fg])

    pca_features_rem = minmax_scale(pca_features_rem)

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_rem

    pca_features_rgb = pca_features_rgb.reshape(B, H, W, 3)*255

    return pca_features_rgb

def pca_map(target_features):
    """
    Generates a 2D grayscale PCA map from 2D target features.
    It applies PCA, normalizes the first principal component to a 0-1 range,
    and then scales it to 0-255 for a grayscale image representation.

    Args:
        target_features (torch.Tensor): A 4D tensor of shape (B, C, H, W)
                                        representing batch, channels, height, width.

    Returns:
        np.ndarray: A 2D NumPy array representing the concatenated grayscale PCA map
                    across the batch, with pixel values scaled to 0-255.
    """
    B, _, H, W = target_features.shape

    input_tensor = target_features.detach().cpu() 
    features = rearrange(input_tensor, 'b d fh fw -> (b fh fw) d').numpy()

    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                        (pca_features[:, 0].max() - pca_features[:, 0].min())
    pca_features = pca_features.reshape(B, H, W, 3)*255

    return np.concatenate([i for i in pca_features], 1)[:,:,0]
