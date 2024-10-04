import time
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import viser, random
import OpenEXR as exr
import Imath, glob, json
from PIL import Image
import scipy
import cv2, os

def read_depth_exr(path):
    exrfile = exr.InputFile(path)
    raw_bytes = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map_r = np.reshape(depth_vector, (height, width))
    return depth_map_r

def depth2pt(depths, rgbs, K, R, xyz_images=False, device = 'cpu'):

    batch_size, h, w = depths.shape
    fx = K[:, 0, 0][:, None, None]
    fy = K[:, 1, 1][:, None, None]
    x_offset = K[:, 0, 2][:, None, None]
    y_offset = K[:, 1, 2][:, None, None]
    indices = np.stack(np.meshgrid(np.arange(batch_size), np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    z_e = depths
    x_e = (indices[..., -1] - x_offset) * z_e / fx
    y_e = (indices[..., -2] - y_offset) * z_e / fy
    homogenerous = np.ones((batch_size, h, w))
    xyz_img = np.stack([x_e, y_e, z_e, homogenerous], axis=-1)  # Shape: [n, H, W, 4]
    xyz_img_trans = np.stack([np.matmul(R[i], xyz_img[i].reshape(-1, 4).T).T.reshape(h, w, 4) for i in range(R.shape[0])], axis=0)
    if xyz_images:
        return xyz_img_trans[..., :3]
    else:
        batch_sign = np.zeros((batch_size, h, w))
        for i in range(batch_size):
            batch_sign[i] = i + 1
        zero_filter = (depths != 0).reshape(-1)
        batch_sign = batch_sign.reshape(-1)[zero_filter]
        return xyz_img_trans[..., :3].reshape(-1, 3)[zero_filter], rgbs[...].reshape(-1,4)[zero_filter], batch_sign, zero_filter

def process_scene(path):

    all_depth_paths = sorted(glob.glob(f'{path}/depth_exr/*'))
    all_rgb_paths = sorted(glob.glob(f'{path}/RGB/*'))

    with open(f'{path}/scene3d.metadata.json','r') as f:
        json_dict = json.load(f)
    RT_matrices = [np.array(cam_pose['rotation']) for cam_pose in json_dict['camera']['poses']]
    K_matrices = [np.array(json_dict['camera']['K'])]*len(RT_matrices)


    depths, Ks, RTs = [], [], []

    hw_size = 256
    stride = 256/hw_size

    cam_wxyz = []
    cam_pos = []
    bboxs = []

    for obj in json_dict['objects']:
        if '-' in os.path.basename(obj['path']):
            bboxs.append(obj['bbox'])

    for idx in range(len(RT_matrices)):
        K_mat = K_matrices[idx].copy()
        RT_mat = RT_matrices[idx]
        depth_path = all_depth_paths[idx]

        RT_mat_inv  = np.linalg.inv(RT_mat)

        quaternion = scipy.spatial.transform.Rotation.from_matrix(RT_mat_inv[:3,:3]).as_quat()
        quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        cam_wxyz.append(quat_wxyz)
        cam_pos.append(RT_mat_inv[:3,3])

        depth = read_depth_exr(depth_path).copy()
        depth_max = depth.max()
        depth[depth == depth_max] = 0

        depth = cv2.resize(depth, (hw_size, hw_size),  interpolation = cv2.INTER_NEAREST)

        K_mat[:2] = K_mat[:2] / stride

        depth = depth
        depths.append(depth)
        Ks.append(K_mat)
        RTs.append(np.linalg.inv(RT_mat))

    images = np.array([np.array(Image.open(i)) for i in all_rgb_paths])

    pnts = depth2pt(np.array(depths), images, np.array(Ks), np.array(RTs))

    point3d = np.array(pnts[0])
    rgb3d = np.array(pnts[1]).astype('uint8')[:,:3]

    cam_K = [K_mat[0,2], K_mat[1,2], K_mat[0,0],K_mat[1,1]]

    return point3d, rgb3d, images, np.array(bboxs), cam_wxyz, cam_pos, cam_K


def draw_bbox(server, bbox, color, line_width, voxel_id):
    cx, cy, cz, h, w, d = bbox  # d is depth

    half_h = h / 2
    half_w = w / 2
    half_d = d / 2

    positions = np.array([
        [-half_w, -half_h, -half_d], [half_w, -half_h, -half_d], 
        [-half_w, half_h, -half_d], [half_w, half_h, -half_d],
        [-half_w, -half_h, half_d], [half_w, -half_h, half_d], 
        [-half_w, half_h, half_d], [half_w, half_h, half_d]
    ]) + np.array([cx, cy, half_d+0.02])

    lines = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3],  # Bottom face
        [4, 5], [4, 6], [5, 7], [6, 7],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ])

    for i in range(len(lines)):
        server.scene.add_spline_catmull_rom(
            f"/line_{voxel_id}_{i}",    # Unique ID for each line
            positions[lines[i]],        # Start and end points of the line
            tension=1.8,                # Tension for the Catmull-Rom spline
            line_width=line_width,             # Width of the line
            color=color,                # Color of the line
            segments=200                # Number of segments
        )


def main(path, port) -> None:

    server = viser.ViserServer(port = port)

    points, colors, images, bboxs, cam_wxyz, cam_pos, cam_K = process_scene(path)

    random_idx = np.arange(len(cam_wxyz))

    gui_point_size = server.gui.add_slider(
        "point size", min=0.0005, max=0.02, step=0.0001, initial_value=0.005
    )
    gui_points = server.gui.add_slider(
        "prune radius",
        min=0.5,
        max=5,
        step=0.1,
        initial_value=2,
    )
    gui_perctge = server.gui.add_slider(
        "prune percentage",
        min=0.005,
        max=1.0,
        step=0.01,
        initial_value=0.9,
    )

    gui_frames = server.gui.add_slider(
        "max camera",
        min=0,
        max=len(images),
        step=1,
        initial_value=min(len(images), 5),
    )
    scale_frames = server.gui.add_slider(
        "scale frames",
        min=0.05,
        max=1,
        step=0.01,
        initial_value=0.15,
    )
    bbox_width = server.gui.add_slider(
        "bbox width",
        min=0,
        max=5,
        step=1,
        initial_value=2,
    )
    point_mask = np.sqrt((points**2).sum(1))<gui_points.value # (points<gui_points.value).all(1) & (points>-gui_points.value).all(1) 
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []
    random.shuffle(random_idx)
    def draw_bbox_scene():
        for bbox_id, bbox_ in enumerate(bboxs):
            draw_bbox(server, bbox_.reshape(-1), np.array([1.0, 0, 0]), bbox_width.value, bbox_id)
    
    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        for frame in frames:
            frame.remove()
        frames.clear()

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position


        for img_id in random_idx[:gui_frames.value]:

            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=cam_wxyz[img_id],
                position=cam_pos[img_id],
                axes_length=0.0,
                axes_radius=0.00,
            )
            frames.append(frame)


            H, W = cam_K[0]*2, cam_K[1]*2
            fy = cam_K[3]

            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=scale_frames.value,
                image=images[img_id],
            )
            attach_callback(frustum, frame)

    draw_bbox_scene()
    
    need_update = True
    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], int(points.shape[0]*(gui_perctge.value)), replace=False)
        points_2 = points[point_mask]
        colors_2 = colors[point_mask]

        point_mask = np.sqrt((points_2**2).sum(1))<gui_points.value
        point_cloud.points = points_2[point_mask]
        point_cloud.colors = colors_2[point_mask]

    @bbox_width.on_update
    def _(_) -> None:
        nonlocal need_update
        draw_bbox_scene()
        need_update = True
        
    @scale_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True


    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        random.shuffle(random_idx)
        need_update = True

    @gui_perctge.on_update
    def _(_) -> None:
        nonlocal need_update
        point_mask = np.random.choice(points.shape[0], int(points.shape[0]*(gui_perctge.value)), replace=False)
        points_2 = points[point_mask]
        colors_2 = colors[point_mask]

        point_mask = np.sqrt((points_2**2).sum(1))<gui_points.value
        point_cloud.points = points_2[point_mask]
        point_cloud.colors = colors_2[point_mask]

        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--scene_path', type=str)
    parser.add_argument('--port', type = int, default = 8000)

    args = parser.parse_args()
    main(path = args.scene_path, port = args.port)
