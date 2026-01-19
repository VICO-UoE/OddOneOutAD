# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
import torch, math
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage, spatial # Added spatial for cdist

import matplotlib.pyplot as plt
import numpy as np
import math


def draw_3d_bboxes(img, bboxes_3d, P, color=(0,0,255), thickness=2, padding=0.0):
    """
    Draws multiple 3D bounding boxes, defined in center format (cx, cy, cz, h, w, d),
    projected onto a 2D image using a given projection matrix.

    Args:
        img (np.ndarray): The input image (H, W, 3) in BGR format or uint8.
        bboxes_3d (list of array-like): A list of 3D bounding boxes, where each box is
                                         represented as [cx, cy, cz, width, height, depth].
        P (np.ndarray): The 3x4 projection matrix used to project 3D points to 2D.
        color (tuple, optional): The BGR color to draw the bounding box edges. Defaults to (0, 0, 255) (red).
        thickness (int, optional): The thickness of the lines used to draw the bounding box edges. Defaults to 2.
        padding (float or tuple, optional): A padding factor applied to the dimensions (height, width, depth)
                                            of each bounding box. Can be a single float for uniform padding
                                            or a tuple (ph, pw, pd) for individual padding factors. Defaults to 0.0.

    Returns:
        np.ndarray: The image with the 3D bounding boxes drawn on it.
    """
    # Ensure uint8 and 3 channels
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ascontiguousarray(img)

    for bbox_3d_c in bboxes_3d:
        cx, cy, cz, w, h, d = bbox_3d_c

        # Apply padding
        if isinstance(padding, (int, float)):
            ph = pw = pd = padding
        else:
            ph, pw, pd = padding

        h = h * (1 + ph)
        w = w * (1 + pw)
        d = d * (1 + pd)

        # Compute corners
        x_min, x_max = cx - w/2, cx + w/2
        y_min, y_max = cy - h/2, cy + h/2
        z_min, z_max = cz - d/2, cz + d/2

        corners_3d = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max]
        ])

        # Project corners
        corners_3d_h = np.hstack([corners_3d, np.ones((8,1))])
        corners_2d_h = (P @ corners_3d_h.T).T
        corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:3]

        # Box edges
        edges = [
            (0,1), (0,2), (0,4),
            (1,3), (1,5),
            (2,3), (2,6),
            (3,7),
            (4,5), (4,6),
            (5,7),
            (6,7)
        ]

        # Draw edges
        for i,j in edges:
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[j].astype(int))
            cv2.line(img, pt1, pt2, color, thickness)

    return img


def visualize_segmentation(image, mask, alpha=0.5, cmap="tab20"):
    """
    Overlays a segmentation mask onto an input image with a specified transparency.

    Args:
        image (np.ndarray): The input image, expected to be a NumPy array of shape [H, W, 3].
                            Pixel values can be in the range 0-1 or 0-255.
        mask (np.ndarray): The segmentation mask, expected to be a NumPy array of shape [H, W]
                           containing integer labels for each segment.
        alpha (float, optional): The transparency factor for the mask overlay. A value of 0.0
                                 means the mask is fully transparent (only image visible),
                                 and 1.0 means the mask is fully opaque (only mask visible).
                                 Defaults to 0.5.
        cmap (str, optional): The colormap to use for mapping integer labels in the mask to colors.
                              Defaults to "tab20".

    Returns:
        np.ndarray: The blended image as a NumPy array of shape [H, W, 3] with float32 pixel
                    values in the range 0-1.
    """
    # normalize image to 0-1 if needed
    if image.max() > 1:
        image_norm = image.astype(np.float32) / 255.0
    else:
        image_norm = image.astype(np.float32)
    
    # map mask to colors
    colormap = plt.get_cmap(cmap, int(mask.max())+1)
    mask_color = colormap(mask)[:, :, :3]  # drop alpha channel
    
    # blend image and mask
    blended = (1 - alpha) * image_norm + alpha * mask_color
    blended = np.clip(blended, 0, 1)
    
    return blended

def save_image_grid(images, filename=None, n_cols=4, cmap=None):
    """
    Saves a list of NumPy images as a grid to a file and returns the grid image as a NumPy array.

    Args:
        images (list of np.ndarray): A list of images, where each image can be
                                     [H, W] (grayscale/single channel) or [H, W, 3] (RGB).
        filename (str, optional): The path to save the grid image. If None, the image is not saved to disk.
                                  Defaults to None.
        n_cols (int, optional): The number of columns in the image grid. Defaults to 4.
        cmap (str, optional): The colormap to use for grayscale or categorical images.
                              If None, "gray" is used for 2D images. Defaults to None.

    Returns:
        np.ndarray: The generated grid image as a NumPy array of shape [H_grid, W_grid, 3] with uint8 pixel values.
    """
    n_imgs = len(images)
    n_rows = math.ceil(n_imgs / n_cols)

    # get aspect ratio from first image
    h, w = images[0].shape[:2]
    aspect = w / h

    # scale figsize according to aspect
    fig_w = n_cols * 3 * aspect
    fig_h = n_rows * 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for idx, ax in enumerate(axes.flat):
        if idx < n_imgs:
            img = images[idx]
            if img.ndim == 2:   # grayscale / label map
                ax.imshow(img, cmap=cmap if cmap else "gray")
            else:               # RGB
                ax.imshow(img)
        ax.axis("off")

    #plt.subplots_adjust(wspace=0, hspace=0)  # no space between images
    plt.tight_layout()
    # save if filename is given
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)

    # convert figure to numpy array
    fig.canvas.draw()
    grid_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_img = grid_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return grid_img


def overlay_images(base_img, overlay_img, alpha=0.5):
    """
    Overlays an `overlay_img` on top of a `base_img` with a specified transparency.
    The `overlay_img` is resized to match the dimensions of the `base_img` before blending.

    Args:
        base_img (np.ndarray): The background image, expected as a NumPy array of shape [H, W, 3].
                               Pixel values can be in the range 0-1 or 0-255.
        overlay_img (np.ndarray): The image to be overlaid, expected as a NumPy array of shape [h, w, 3].
                                  Pixel values can be in the range 0-1 or 0-255.
        alpha (float, optional): The transparency factor for the overlay. A value of 0.0 means
                                 only the base image is visible, and 1.0 means only the overlay
                                 image is visible. Defaults to 0.5.

    Returns:
        np.ndarray: The blended image as a NumPy array of shape [H, W, 3] with float32 pixel
                    values in the range 0-1.
    """
    H, W, _ = base_img.shape

    # resize overlay to match base
    overlay_resized = cv2.resize(overlay_img, (W, H), interpolation=cv2.INTER_NEAREST)

    # normalize to 0–1 if needed
    if overlay_resized.max() > 1:
        overlay_resized = overlay_resized.astype(np.float32) / 255.0
    if base_img.max() > 1:
        base_norm = base_img.astype(np.float32) / 255.0
    else:
        base_norm = base_img.astype(np.float32)

    blended = (1 - alpha) * base_norm + alpha * overlay_resized
    blended = np.clip(blended, 0, 1)

    return blended

def process_3d_mask_volume(volume, threshold = 0.3, num_cluster = None, sparsity_constant = 1.5, max_clusters_to_check = 20):
    
    """
    Processes a 3D mask volume to identify and segment distinct objects by applying
    thresholding, 2D projection, spectral clustering, and post-processing steps.

    Args:
        volume (torch.Tensor): The input 3D volume tensor, typically representing density or features.
                               Expected shape: (D, H, W) or similar.
        threshold (float, optional): The density threshold used to create an initial binary mask
                                     from the input volume. Defaults to 0.3.
        num_cluster (int, optional): The desired number of clusters for spectral clustering.
                                     If None, the optimal number of clusters is estimated
                                     using the eigengap heuristic. Defaults to None.
        sparsity_constant (float, optional): A scaling factor applied to the mean similarity
                                              when constructing the affinity matrix for spectral clustering.
                                              Defaults to 1.4.
        max_clusters_to_check (int, optional): The maximum number of eigenvalues to consider
                                               when estimating the optimal number of clusters.
                                               Defaults to 20.

    Returns:
        tuple[np.ndarray, list]: A tuple containing:
            - merged_labeled_mask (np.ndarray): A 3D labeled mask where each distinct object
                                                is assigned a unique integer label (1 to N).
            - bounding_boxes_list (list): A list of bounding boxes for each segmented object,
                                          where each bounding box is represented as
                                          [min_z, min_y, min_x, max_z, max_y, max_x].
    """

    # --- 1. Initial Masking and 2D Projection ---
    # Convert volume to numpy array and apply threshold to get a 3D mask.
    density = volume.cpu().detach().numpy()


    # Project the 3D density to a 2D image by summing along the first axis.
    dens2d_im = density.sum(0) 
    
    intial_thr = threshold
    # Iteratively reduce the threshold until a sufficient number of 2D points are found.
    for _ in range(100):
        intial_thr = intial_thr * 0.9
        pnts2d_2 = np.argwhere(dens2d_im > intial_thr)
        if len(pnts2d_2) > 5:
            break

    mask = density > intial_thr
    coords = np.argwhere(mask)

    # --- 2. Spectral Clustering for 2D Points ---
    # Inlined logic from perform_spectral_clustering_with_centers
    points = torch.Tensor(pnts2d_2).cuda()

    device = points.device
    points_gpu = points.float().to(device)
    N = points_gpu.shape[0]

    # Compute Affinity Matrix, Laplacian, and Eigendecomposition
    dists = torch.cdist(points_gpu, points_gpu)
    sigma = torch.mean(dists)
    S = torch.exp(-dists**2 / (2 * sigma**2))
    mean_similarity = torch.mean(S)
    W = torch.where(S >= sparsity_constant * mean_similarity, S, torch.tensor(0.0, device=device))
    D = torch.diag(torch.sum(W, dim=1))
    D_diag_vals = torch.diag(D)
    D_inv_sqrt_vals = 1.0 / torch.sqrt(torch.where(D_diag_vals > 0, D_diag_vals, torch.tensor(1e-8, device=device)))
    D_inv_sqrt = torch.diag(D_inv_sqrt_vals)
    I = torch.eye(N, device=device)
    L_norm = I - torch.matmul(torch.matmul(D_inv_sqrt, W), D_inv_sqrt)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)

    # Estimate Optimal k
    if num_cluster is None: 
        eigenvalues_subset = eigenvalues[:max_clusters_to_check]
        eigen_gaps = torch.diff(eigenvalues_subset)
        num_cluster = torch.argmax(eigen_gaps).item() + 1

    # Perform K-Means on Eigenvectors
    U = eigenvectors[:, :num_cluster]
    norm = torch.norm(U, p=2, dim=1, keepdim=True)
    T = U / torch.where(norm > 0, norm, torch.tensor(1e-8, device=device))
    T_np = T.cpu().numpy()
    kmeans = KMeans(n_clusters=num_cluster, n_init='auto', random_state=42)
    labels_spectral = kmeans.fit_predict(T_np) # Renamed to avoid conflict
    
    # Calculate Cluster Centers in Original Space
    points_np = points.cpu().numpy()
    k2d = np.zeros((num_cluster, points_np.shape[1]))
    for i in range(num_cluster):
        points_in_cluster = points_np[labels_spectral == i]
        if len(points_in_cluster) > 0:
            k2d[i] = points_in_cluster.mean(axis=0)
    # End inlined perform_spectral_clustering_with_centers


    # Assign each 3D voxel to the closest 2D cluster center based on its (y, x) coordinates.
    labels = spatial.distance.cdist(coords[:, 1:], k2d).argmin(1)

    # Create a labeled 3D mask where each voxel is assigned a cluster label.
    labeled_mask = np.zeros_like(density, dtype=np.int32)
    labeled_mask[tuple(coords.T)] = labels + 1  # coords.T splits into x,y,z arrays

    # --- 4. Post-processing and Visualization of Individual Objects ---
    vols = []
    bounding_boxes_list = [] # New list to store bounding boxes
    for label in range(1, labeled_mask.max() + 1):
        binary_obj_mask = (labeled_mask == label)

        # Inlined logic from remove_distant_blobs
        mask_to_clean = binary_obj_mask
        min_size = 10
        max_dist = None # Will be set automatically
        
        struct_remove = ndimage.generate_binary_structure(3, 1)
        labeled_blobs, num_blobs = ndimage.label(mask_to_clean, structure=struct_remove)

        if num_blobs == 0:
            cleaned_mask = mask_to_clean
        else:
            coords_all = np.argwhere(mask_to_clean)
            global_centroid = coords_all.mean(axis=0)
            cleaned_mask = np.zeros_like(mask_to_clean, dtype=bool)

            for blob_label in range(1, num_blobs + 1):
                coords_blob = np.argwhere(labeled_blobs == blob_label)
                size_blob = coords_blob.shape[0]
                centroid_blob = coords_blob.mean(axis=0)
                dist_blob = np.linalg.norm(centroid_blob - global_centroid)

                if max_dist is None:
                    max_dist_auto = np.linalg.norm(mask_to_clean.shape) / 2

                if dist_blob >= max_dist_auto and size_blob <= min_size:
                    continue
                
                cleaned_mask[labeled_blobs == blob_label] = True
        # End inlined remove_distant_blobs

        # Inlined logic from binary_dilation_3d
        labeled_mask_to_dilate = cleaned_mask # Use the output from remove_distant_blobs

        
        closed_obj = np.zeros_like(labeled_mask_to_dilate)
        struct_dilate = ndimage.generate_binary_structure(3, 1)

        # Since we are processing a single binary_obj, we can simplify
        closed_obj = ndimage.binary_dilation(labeled_mask_to_dilate, structure=struct_dilate, iterations=1)
        binary_obj = closed_obj
        # End inlined binary_dilation_3d

        vols.append(binary_obj)

        # Calculate bounding box for the current binary_obj
        if binary_obj.any():
            obj_coords = np.argwhere(binary_obj)
            min_coor = obj_coords.min(axis=0)
            max_coor = obj_coords.max(axis=0)
            bounding_boxes_list.append([*min_coor, *max_coor])


    # --- 5. Merge and Final Visualization ---
    # Inlined logic from merge_binary_objects
    merged_labeled_mask = np.zeros_like(vols[0], dtype=np.int32)
    current_label = 1
    for v in vols:
        if v.any():
            merged_labeled_mask[v] = current_label
            current_label += 1
    # End inlined merge_binary_objects

    return merged_labeled_mask, bounding_boxes_list # Return bounding boxes as well


def perform_spectral_clustering_with_centers(points, max_clusters_to_check=11, sparsity_constant=1.4):
    """
    Estimates the optimal number of clusters (k) using the eigengap heuristic,
    performs spectral clustering on the input points, and returns the cluster labels
    and their centers in the original feature space.

    Args:
        points (torch.Tensor): A tensor of shape (N, D) representing N data points
                               in D-dimensional space.
        max_clusters_to_check (int, optional): The maximum number of eigenvalues to consider
                                               when estimating the optimal number of clusters (k).
                                               Defaults to 11.
        sparsity_constant (float, optional): A scaling factor applied to the mean similarity
                                              when constructing the affinity matrix for spectral clustering.
                                              Defaults to 1.4.

    Returns:
        tuple[int, np.ndarray, np.ndarray]: A tuple containing:
            - optimal_k (int): The estimated optimal number of clusters.
            - labels (np.ndarray): A NumPy array of shape (N,) containing the cluster label
                                   (an integer from 0 to optimal_k-1) for each input point.
            - centers (np.ndarray): A NumPy array of shape (optimal_k, D) containing the
                                    coordinates of the cluster centers in the original
                                    D-dimensional space.
    """
    device = points.device
    points_gpu = points.float().to(device)
    N = points_gpu.shape[0]

    # --- Step 1 & 2: Affinity Matrix, Laplacian, and Eigendecomposition ---
    # (This part is identical to the previous function)
    dists = torch.cdist(points_gpu, points_gpu)
    sigma = torch.mean(dists)
    S = torch.exp(-dists**2 / (2 * sigma**2))
    mean_similarity = torch.mean(S)

    W = torch.where(S >= sparsity_constant * mean_similarity, S, torch.tensor(0.0, device=device))
    D = torch.diag(torch.sum(W, dim=1))
    D_diag_vals = torch.diag(D)
    D_inv_sqrt_vals = 1.0 / torch.sqrt(torch.where(D_diag_vals > 0, D_diag_vals, torch.tensor(1e-8, device=device)))
    D_inv_sqrt = torch.diag(D_inv_sqrt_vals)
    I = torch.eye(N, device=device)
    L_norm = I - torch.matmul(torch.matmul(D_inv_sqrt, W), D_inv_sqrt)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)

    # --- Step 3: Estimate Optimal k ---
    eigenvalues_subset = eigenvalues[:max_clusters_to_check]
    eigen_gaps = torch.diff(eigenvalues_subset)
    optimal_k = torch.argmax(eigen_gaps).item() + 1

    # --- Step 4: Perform K-Means on Eigenvectors ---
    U = eigenvectors[:, :optimal_k]
    norm = torch.norm(U, p=2, dim=1, keepdim=True)
    T = U / torch.where(norm > 0, norm, torch.tensor(1e-8, device=device))
    T_np = T.cpu().numpy()
    kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(T_np)
    
    # --- Step 5: Calculate Cluster Centers in Original Space ---
    # This is the new part
    points_np = points.cpu().numpy()
    centers = np.zeros((optimal_k, points_np.shape[1]))
    for i in range(optimal_k):
        # Find all points belonging to the current cluster
        points_in_cluster = points_np[labels == i]
        # Compute the mean of these points to find the center
        if len(points_in_cluster) > 0:
            centers[i] = points_in_cluster.mean(axis=0)

    return optimal_k, labels, centers

def merge_binary_objects(vols):
    """
    Merges a list of binary 3D masks into a single labeled mask.
    Each non-empty binary mask in the input list is assigned a unique integer label
    (starting from 1) in the output merged mask.

    Args:
        vols (list of np.ndarray): A list of 3D binary NumPy arrays, where each array
                                   represents a binary mask of an object.

    Returns:
        np.ndarray: A 3D NumPy array of the same shape as the input masks, where
                    each object is assigned a unique integer label. Background is 0.
    """
    labeled_mask = np.zeros_like(vols[0], dtype=np.int32)
    current_label = 1
    for v in vols:
        if v.any():
            labeled_mask[v] = current_label
            current_label += 1
    return labeled_mask

def remove_distant_blobs(mask, min_size=10, max_dist=None):
    """
    Removes small or distant blobs (connected components) from a 3D binary mask.
    Blobs are considered "distant" if their centroid is far from the global centroid
    of all masked points and their size is below `min_size`.

    Args:
        mask (np.ndarray): A 3D binary NumPy array representing the mask.
                           True values indicate foreground, False for background.
        min_size (int, optional): The minimum number of voxels a blob must have to be considered
                                  significant. Blobs smaller than this might be removed if distant.
                                  Defaults to 10.
        max_dist (float, optional): The maximum allowed Euclidean distance from a blob's centroid
                                    to the global centroid. If a blob's centroid is further than
                                    `max_dist` AND its size is less than or equal to `min_size`,
                                    it will be removed. If None, `max_dist` is automatically
                                    estimated as half the diagonal length of the mask's shape.
                                    Defaults to None.

    Returns:
        np.ndarray: A cleaned 3D binary NumPy array with small/distant blobs removed.
    """
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected in 3D
    labeled, num = ndimage.label(mask, structure=struct)

    if num == 0:
        return mask

    # --- Global centroid ---
    coords = np.argwhere(mask)
    global_centroid = coords.mean(axis=0)

    # --- Distances for each blob ---
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, num + 1):
        coords = np.argwhere(labeled == label)
        size = coords.shape[0]
        # if size < min_size:
        #     continue

        centroid = coords.mean(axis=0)
        dist = np.linalg.norm(centroid - global_centroid)

        # automatic distance threshold if not given
        if max_dist is None:
            max_dist = np.linalg.norm(mask.shape) / 2  # heuristic: 1/4 diagonal

        if dist >= max_dist and size <= min_size:
            continue
        
        cleaned[labeled == label] = True

    return cleaned


def binary_dilation_3d(labeled_mask, iterations=1):
    """
    Applies morphological dilation to each individual object (label) within a 3D labeled mask.
    This operation expands the boundaries of foreground regions.

    Args:
        labeled_mask (np.ndarray): A 3D NumPy array representing the labeled mask, where
                                   0 is background and positive integers are object labels.
        iterations (int, optional): The number of times the dilation operation is applied.
                                    Defaults to 1.

    Returns:
        np.ndarray: A 3D NumPy array of the same shape as `labeled_mask`, with dilation
                    applied to each object.
    """
    closed = np.zeros_like(labeled_mask)
    struct = ndimage.generate_binary_structure(3, 1)  # 3D, 6-connected

    for label in range(1, labeled_mask.max() + 1):
        binary_obj = (labeled_mask == label)

        closed_obj = ndimage.binary_dilation(binary_obj, structure=struct, iterations=iterations)

        closed[closed_obj] = label  # assign back

    return closed
    
def merge_small_blobs(labeled_mask, merge_labels):
    """
    Merges specified small or outlier blobs within a 3D labeled mask into their
    nearest neighboring blobs. This is useful for cleaning up segmentation results.

    Args:
        labeled_mask (np.ndarray): A 3D NumPy array representing the labeled mask,
                                   where 0 is background and positive integers are object labels.
        merge_labels (list): A list of integer labels corresponding to the blobs
                             that should be merged into their nearest neighbors.

    Returns:
        np.ndarray: A 3D NumPy array of the same shape as `labeled_mask`, with the
                    specified blobs merged into their nearest neighbors.
    """
    merged_mask = labeled_mask.copy()
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    for label_id in merge_labels:
        # mask for the small/outlier blob
        blob_mask = (merged_mask == label_id)
        if blob_mask.sum() == 0:
            continue

        # dilate blob until it touches neighbors
        dilated = blob_mask.copy()
        neighbors = np.unique(merged_mask[dilated])
        neighbors = neighbors[(neighbors != 0) & (neighbors != label_id)]

        # keep dilating until we find neighbors
        max_iter = 10  # safety to prevent infinite loops
        iter_count = 0
        while len(neighbors) == 0 and iter_count < max_iter:
            dilated = ndimage.binary_dilation(dilated, structure=structure)
            neighbors = np.unique(merged_mask[dilated])
            neighbors = neighbors[(neighbors != 0) & (neighbors != label_id)]
            iter_count += 1

        if len(neighbors) > 0:
            # merge into the nearest neighbor (by label voxel count)
            neighbor_sizes = [np.sum(merged_mask == n) for n in neighbors]
            nearest_neighbor = neighbors[np.argmax(neighbor_sizes)]
            merged_mask[blob_mask] = nearest_neighbor
        # else: leave blob alone if no neighbor found

    return merged_mask


def visualize_mask(labeled_mask):
    """
    Visualizes a 3D labeled mask by projecting it to 2D (taking the maximum label along the first axis)
    and assigning distinct colors to each unique label.

    Args:
        labeled_mask (np.ndarray): A 3D NumPy array representing the labeled mask,
                                   where 0 is background and positive integers are object labels.

    Returns:
        np.ndarray: A 2D NumPy array of shape (H, W, 3) with uint8 pixel values,
                    representing the colored 2D projection of the labeled mask.
    """
    labels_2d = labeled_mask.max(0)
        
        # Find unique labels in the mask
    labels = np.unique(labels_2d)
    num_labels = len(labels)

    # Generate distinct colors for each label
    # Here we use a simple colormap: evenly spaced in HSV -> RGB
    colors = plt.cm.get_cmap('tab20', num_labels)  # 'tab20' gives up to 20 distinct colors

    # Map label to color
    h, w = labels_2d.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, label in enumerate(labels):
        rgb = (np.array(colors(i)[:3]) * 255).astype(np.uint8)  # convert to 0-255
        colored_mask[labels_2d == label] = rgb

    return colored_mask


def sor_filter(points, k=20, std_ratio=2.0):
    """
    Applies Statistical Outlier Removal (SOR) to a 3D point cloud.
    This method filters out points that are statistically distant from their neighbors.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing N 3D points.
        k (int, optional): The number of nearest neighbors to consider for each point
                           when calculating mean distances. Defaults to 20.
        std_ratio (float, optional): A threshold multiplier for the standard deviation.
                                     Points whose mean distance to neighbors is outside
                                     `mean +/- std_ratio * std` will be considered outliers.
                                     Defaults to 2.0.

    Returns:
        np.ndarray: A NumPy array containing only the inlier points after outlier removal.
    """
    # Nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=min(len(points),k+1)).fit(points)  
    dists, _ = nbrs.kneighbors(points)
    
    # Ignore the first column (distance to itself = 0)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    
    # Global statistics
    mean = np.mean(mean_dists)
    std = np.std(mean_dists)
    
    # Keep points within threshold
    mask = np.abs(mean_dists - mean) <= std_ratio * std
    return points[mask]


import torch
from pytorch3d.structures import Pointclouds

def downsample_points_p3d(points, max_points=3000, voxel_size=None):
    """
    Downsamples a 3D point cloud using a voxel-grid approach.
    If the number of points is already below `max_points`, the original points are returned.
    Otherwise, a voxel grid is created, and points within each voxel are averaged to
    produce a downsampled point cloud.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) representing N 3D points on the GPU.
        max_points (int, optional): The target maximum number of points in the downsampled
                                    point cloud. If `voxel_size` is None, this is used
                                    to automatically compute an appropriate voxel size.
                                    Defaults to 3000.
        voxel_size (float, optional): The size of the voxels in the grid. If None, the
                                      voxel size is automatically computed based on the
                                      bounding box of the points and `max_points`.
                                      Defaults to None.

    Returns:
        torch.Tensor: The downsampled point cloud as a tensor of shape (M, 3), where M
                      is less than or equal to `max_points`.
    """
    device = points.device
    N = points.shape[0]

    # Early return if already under max_points
    if N <= max_points:
        return points

    # Automatically compute voxel size if not provided
    if voxel_size is None:
        bbox = points.max(0).values - points.min(0).values
        voxel_size = (bbox.max() / (max_points ** (1/3))).item()

    # Convert to PyTorch3D Pointclouds
    p3d_pc = Pointclouds(points=[points])

    # Voxel-grid subsampling
    # PyTorch3D does not have a direct function, so we use unique voxel trick
    coords = points / voxel_size
    voxel_idx, inverse = torch.unique(torch.floor(coords).to(torch.int32), dim=0, return_inverse=True)
    
    # Average points inside each voxel
    downsampled = torch.zeros((voxel_idx.shape[0], 3), device=device)
    downsampled.index_add_(0, inverse, points)
    counts = torch.bincount(inverse, minlength=voxel_idx.shape[0]).float().unsqueeze(1)
    downsampled /= counts

    return downsampled


def dbscan_clustering(point_cloud, min_samples=10):
    """
    Performs HDBSCAN clustering on a 3D point cloud.
    Note: The clustering is applied to the Y and Z coordinates (columns 1 and 2) of the point cloud.

    Args:
        point_cloud (np.ndarray): A NumPy array of shape (N, 3) representing N 3D points.
        min_samples (int, optional): The minimum number of samples in a neighborhood for a point
                                     to be considered as a core point. Also, the smallest size
                                     grouping for which we would consider a cluster. Defaults to 10.

    Returns:
        np.ndarray: A NumPy array of shape (N,) containing the cluster label for each point.
                    Noise points are assigned the label -1.
    """
    point_cloud_ = point_cloud.copy()
    clustering = HDBSCAN(min_cluster_size=min_samples).fit(point_cloud_[:,1:])
    labels = clustering.labels_

    return labels


def corner_to_target(corner_boxes, voxel_size, h_vox, w_vox, d_vox):
    """
    Converts 3D bounding boxes from corner format (z_min, y_min, x_min, z_max, y_max, x_max)
    in voxel coordinates to center format (xc, yc, zc, h, w, l) in original world coordinates.

    Args:
        corner_boxes (torch.Tensor): A tensor of shape (N, 6) where each row represents
                                     a bounding box in corner format [z_min, y_min, x_min, z_max, y_max, x_max]
                                     in voxel coordinates.
        voxel_size (float): The size of a single voxel in world units.
        h_vox (int): The height of the voxel grid.
        w_vox (int): The width of the voxel grid.
        d_vox (int): The depth of the voxel grid.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) where each row represents a bounding box
                      in center format [xc, yc, zc, height, width, length] in original
                      world coordinates.
    """
    zmin, ymin, xmin, zmax, ymax, xmax = corner_boxes.T
    xc, yc, zc = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
    w, h, l = xmax - xmin, ymax - ymin, zmax - zmin
    boxes = torch.stack([xc, yc, zc, h, w, l], dim=1)
    boxes[:, :3] -= torch.tensor([h_vox/2, w_vox/2, d_vox/2], device=boxes.device)
    return boxes * voxel_size

def target_to_corner(target_boxes, voxel_size, h_vox, w_vox, d_vox):
    """
    Converts 3D bounding boxes from center format (xc, yc, zc, h, w, l) in original
    world coordinates to corner format (z_min, y_min, x_min, z_max, y_max, x_max)
    in voxel coordinates.

    Args:
        target_boxes (torch.Tensor): A tensor of shape (N, 6) where each row represents
                                     a bounding box in center format [xc, yc, zc, height, width, length]
                                     in original world coordinates.
        voxel_size (float): The size of a single voxel in world units.
        h_vox (int): The height of the voxel grid.
        w_vox (int): The width of the voxel grid.
        d_vox (int): The depth of the voxel grid.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) where each row represents a bounding box
                      in corner format [z_min, y_min, x_min, z_max, y_max, x_max] in
                      voxel coordinates.
    """
    boxes = target_boxes / voxel_size
    boxes[:, :3] += torch.tensor([h_vox/2, w_vox/2, d_vox/2], device=boxes.device)
    xc, yc, zc, h, w, l = boxes.T
    return torch.stack([zc - l/2, yc - h/2, xc - w/2,
                        zc + l/2, yc + h/2, xc + w/2], dim=1)

def extract_bounding_boxes(points, labels):
    """
    Extracts the axis-aligned bounding box for each cluster in a point cloud.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing N 3D points.
        labels (np.ndarray): A NumPy array of shape (N,) containing the cluster label
                             for each point. Noise points (label -1) are ignored.

    Returns:
        dict: A dictionary where keys are cluster labels (integers) and values are
              tuples representing the bounding box for that cluster in
              (min_x, min_y, min_z, max_x, max_y, max_z) format.
    """
    bounding_boxes = {}
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = points[labels == label]

        if len(cluster_points) > 0:
            min_vals_cluster = cluster_points.min(axis=0)
            max_vals_cluster = cluster_points.max(axis=0)
            bounding_boxes[label] = (*min_vals_cluster, *max_vals_cluster)
    
    return bounding_boxes

def save_point_cloud_with_clusters(points, labels, filename="clustered_point_cloud.ply"):
    """
    Saves a 3D point cloud with cluster labels to a .ply file, where each cluster
    is assigned a distinct color. Noise points (label -1) are colored black.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing N 3D points.
        labels (np.ndarray): A NumPy array of shape (N,) containing the cluster label
                             for each point.
        filename (str, optional): The name of the output PLY file. Defaults to "clustered_point_cloud.ply".
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    print(f"Detected {num_clusters} clusters (including noise).")

    # Generate random colors for each cluster
    # Ensure consistent colors
    colors = np.random.rand(len(unique_labels), 3)  # Random RGB colors

    # Assign colors: black for noise (-1), unique color per cluster
    colored_points = np.array([colors[label] if label != -1 else [0, 0, 0] for label in labels])

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colored_points)

    # Save as PLY
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved clustered point cloud to '{filename}'.")
    
def center_to_corner(boxes):
    """
    Converts 3D bounding boxes from center format (xc, yc, zc, h, w, l) to
    corner format (x_min, y_min, z_min, x_max, y_max, z_max).

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 6) where each row represents
                              a bounding box in center format [xc, yc, zc, height, width, length].

    Returns:
        torch.Tensor: A tensor of shape (N, 6) where each row represents a bounding box
                      in corner format [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    xc, yc, zc, h, w, l = boxes.unbind(-1)
    x_min = xc - w / 2
    x_max = xc + w / 2
    y_min = yc - h / 2
    y_max = yc + h / 2
    z_min = zc - l / 2
    z_max = zc + l / 2
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=-1)


# ----------------------------
# 1. Compute 3D IoU
# ----------------------------
def compute_iou_3d(boxes1, boxes2):
    """
    Computes the 3D Intersection over Union (IoU) between two sets of 3D bounding boxes.

    Args:
        boxes1 (torch.Tensor): A tensor of shape (N, 6) representing N bounding boxes
                               in corner format [x_min, y_min, z_min, x_max, y_max, z_max].
        boxes2 (torch.Tensor): A tensor of shape (M, 6) representing M bounding boxes
                               in corner format [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        torch.Tensor: A tensor of shape (N, M) containing the IoU values for each pair
                      of bounding boxes from `boxes1` and `boxes2`.
    """
    boxes1_expanded = boxes1.unsqueeze(1) # (N, 1, 6)
    boxes2_expanded = boxes2.unsqueeze(0) # (1, M, 6)

    inter_min = torch.max(boxes1_expanded[:, :, :3], boxes2_expanded[:, :, :3])
    inter_max = torch.min(boxes1_expanded[:, :, 3:], boxes2_expanded[:, :, 3:])

    inter_dims = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]

    dims1 = (boxes1[:, 3:] - boxes1[:, :3])
    vol1 = dims1[:, 0] * dims1[:, 1] * dims1[:, 2]
    dims2 = (boxes2[:, 3:] - boxes2[:, :3])
    vol2 = dims2[:, 0] * dims2[:, 1] * dims2[:, 2]

    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol
    iou = inter_vol / (union_vol + 1e-6) # Add small epsilon to avoid division by zero
    return iou

# ----------------------------
# 2. Match predictions and compute AP
# ----------------------------
def compute_ap(bbox_target, bbox_pred):
    """
    Computes Average Precision (AP) for 3D bounding box detection.
    This function currently only computes the IoU matrix between predicted and ground truth boxes.
    The full AP calculation (matching, precision-recall curve, etc.) is not implemented.

    Args:
        bbox_target (dict): A dictionary of ground truth bounding boxes. Keys are IDs, values are box tensors.
        bbox_pred (dict): A dictionary of predicted bounding boxes. Keys are IDs, values are box tensors.

    Returns:
        np.ndarray: A NumPy array of shape (len(bbox_pred), len(bbox_target)) containing
                    the 3D IoU values between all pairs of predicted and ground truth boxes.
    """
    pred_ids = list(bbox_pred.keys())
    pred_boxes = [bbox_pred[i] for i in pred_ids]

    gt_ids = list(bbox_target.keys())
    gt_boxes = [bbox_target[i] for i in gt_ids]

    ious = np.zeros((len(pred_boxes), len(gt_boxes)))

    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            ious[i, j] = compute_iou_3d(pb, gb)
    return ious

def compute_iou(box1: tuple, box2: tuple) -> float:
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): Bounding box 1 in (min_x, min_y, max_x, max_y) format.
        box2 (tuple): Bounding box 2 in (min_x, min_y, max_x, max_y) format.

    Returns:
        float: The IoU value.
    """
    # Unpack the coordinates for each box
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate the area of intersection
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height

    # Calculate the area of each bounding box
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def object_selector(density_3d, targets, visualize = False, thr = 0.5, sparsity_constant = 1.4, voxel_size = 0.04, estimate_number_objs = False):
    """
    Selects objects from a 3D density volume based on predicted masks and computes
    metrics like IoU and cardinality accuracy against ground truth targets.
    Optionally visualizes the segmented masks.

    Args:
        density_3d (torch.Tensor): A 5D tensor representing the 3D density volume.
                                   Expected shape: (B, C, D, H, W), where B is batch size,
                                   C is channels, D is depth, H is height, W is width.
        targets (list of dict): A list of dictionaries, where each dictionary corresponds
                                to a batch item and contains a 'boxes' key with ground truth
                                bounding boxes (torch.Tensor).
        visualize (bool, optional): If True, generates and stores 2D visualizations of
                                    the segmented masks. Defaults to False.
        thr (float, optional): The density threshold used in `process_3d_mask_volume`
                               to create an initial binary mask. Defaults to 0.5.
        voxel_size (float, optional): The size of a single voxel in world units, used
                                      for converting bounding box coordinates. Defaults to 0.04.

    Returns:
        tuple: A tuple containing:
            - predict_bboxes (list of dict): A list of dictionaries, each containing
                                            the predicted bounding boxes for a batch item,
                                            matched to ground truth boxes based on IoU.
            - labeled_masked (np.ndarray): A stacked NumPy array of 3D labeled masks
                                           for all batch items.
            - metrics (list): A list containing:
                - ious (list): Mean IoU for each batch item.
                - cardinality_acc (list): Boolean indicating if the number of predicted
                                          boxes matches the number of ground truth boxes.
                - imgs (list): List of 2D visualization images (NumPy arrays) if `visualize` is True,
                               otherwise None for each item.
    """
    density_3d = density_3d.detach()
    predict_bboxes = []

    B, _, d_vox, h_vox, w_vox = density_3d.shape

    ious = []
    cardinality_acc = []
    imgs = []
    labeled_masked = []

    for i in range(B):

        tgt_boxes_i = targets[i]['boxes']

        num_cluster = None if estimate_number_objs else len(tgt_boxes_i)

        try:
            mask3d, pred_boxes_i = process_3d_mask_volume(density_3d[i,0], num_cluster = num_cluster, threshold = thr, sparsity_constant = sparsity_constant)
        except:     
            cardinality_acc.append(False)
            ious.append(0)
            imgs.append(None)
            labeled_masked.append(np.zeros(density_3d[i,0].shape))
            predict_bboxes.append({'boxes':torch.zeros((0,6))})
            continue

        pred_boxes_i = corner_to_target(torch.Tensor(list(pred_boxes_i)), voxel_size, h_vox, w_vox, d_vox)
        pred_boxes_i = torch.cat([pred_boxes_i[:,:3], pred_boxes_i[:,4:5], pred_boxes_i[:,3:4],  pred_boxes_i[:,5:6]], 1)

        cardinality_acc.append(len(pred_boxes_i) == len(tgt_boxes_i))

        IoU_matrix = compute_iou_3d(center_to_corner(tgt_boxes_i), center_to_corner(pred_boxes_i))

        miou = IoU_matrix.max(0).values.mean()

        predict_bboxes.append({'boxes':pred_boxes_i[IoU_matrix.argmax(1)]})

        if visualize:
            im_array = visualize_mask(mask3d)
        else:
            im_array = None

        ious.append(miou)

        imgs.append(im_array)

        labeled_masked.append(mask3d)

    labeled_masked = np.stack(labeled_masked)

    return predict_bboxes, labeled_masked, [ious, cardinality_acc, imgs]
