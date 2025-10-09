"""
Inference script: Given a trained MPI checkpoint and a dataset sequence,
use two views to build an MPI and render all other views.

Usage:
    python run.py --checkpoint path/to/checkpoint.tar \
                  --dataset_type re10k \
                  --data_path /path/to/re10k \
                  --scene_idx 0 \
                  --ref_idx 0 \
                  --src_idx 1 \
                  --output_dir outputs/
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

from networks import StereoMagnificationModel
from utils import *


parser = ArgumentParser(description="MPI Inference on arbitrary sequences")
parser.add_argument('--checkpoint', required=True, type=str, help="Path to trained checkpoint")
parser.add_argument('--dataset_type', default='re10k', choices=['re10k', 'colmap'], 
                    help="Dataset type (re10k or colmap)")
parser.add_argument('--data_path', required=True, type=str, 
                    help="Path to dataset (for re10k: parent dir of train/test)")
parser.add_argument('--scene_idx', default=0, type=int, 
                    help="Scene index to use (for datasets with multiple scenes)")
parser.add_argument('--ref_idx', default=0, type=int, 
                    help="Reference view index (used to build MPI)")
parser.add_argument('--src_idx', default=1, type=int, 
                    help="Source view index (used to build MPI)")
parser.add_argument('--output_dir', default='outputs/', type=str, 
                    help="Directory to save rendered outputs")
parser.add_argument('--img_size', default=None, type=int, nargs=2,
                    help="Resize images to (W, H). If None, use original size.")
parser.add_argument('--num_planes', default=32, type=int, help="Number of MPI planes")
parser.add_argument('--save_mpi_layers', action='store_true', 
                    help="Save individual MPI layer visualizations")
parser.add_argument('--compute_metrics', action='store_true',
                    help="Compute PSNR/SSIM if ground truth available")


def load_re10k_scene(data_path, scene_idx=0, split='test'):
    """Load a scene from RE10K preprocessed format."""
    shard_dir = os.path.join(data_path, split)
    shards = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir)])
    
    # Load shards until we find the requested scene
    scene_count = 0
    for shard_path in shards:
        try:
            scenes = torch.load(shard_path, weights_only=False)
            if scene_count + len(scenes) > scene_idx:
                scene = scenes[scene_idx - scene_count]
                return scene
            scene_count += len(scenes)
        except Exception as e:
            print(f"Error loading shard {shard_path}: {e}")
            continue
    
    raise ValueError(f"Scene index {scene_idx} not found in dataset")


def prepare_view(image_bytes, camera, img_size, device='cuda', decode=True):
    """Decode and prepare a single view."""
    from torchvision.io import decode_image
    
    # Decode image
    if decode:
        img = decode_image(image_bytes, mode="RGB")  # (C, H, W)
    else:
        img = torch.tensor(image_bytes) # though, these are not image bytes but actual tensors.
    
    # Extract camera params
    fx, fy, cx, cy = camera[0:4]
    Rt_flat = camera[6:]
    
    # Resize if requested
    if img_size is not None:
        W, H = img_size
        scaled_img = F.interpolate(img.unsqueeze(0), (H, W)).squeeze(0)
    else:
        _, H, W = img.shape
        scaled_img = img
    
    # Build intrinsics (denormalize)
    intr = make_intrinsics_matrix(
        W * fx, H * fy, W * cx, H * cy, device=device
    )
    
    # Build extrinsics (w2c)
    extr = to_homogenous(make_extrinsics_matrix(Rt_flat, device=device), device=device)
    
    # Preprocess image to [-1, 1]
    img_tensor = preprocess_image_torch(scaled_img.float() / 255.0).permute(1, 2, 0)  # HWC
    
    return {
        'image': img_tensor.to(device),
        'intrinsics': intr,
        'pose': extr,
    }


def build_mpi(model, ref_view, src_view, num_planes=32, device='cuda'):
    """Build MPI from reference and source views."""
    # Compute plane-sweep volume
    psv_planes = torch.Tensor(inv_depths(1, 100, num_planes)).to(device)
    curr_pose = torch.matmul(src_view['pose'], torch.inverse(ref_view['pose']))
    curr_psv = plane_sweep_torch_one(
        src_view['image'], psv_planes, curr_pose, src_view['intrinsics'], device=device
    )
    
    # Build network input: [ref_image, psv_volume]
    net_input = torch.cat([
        torch.unsqueeze(ref_view['image'], 0), 
        curr_psv
    ], dim=3)  # [1, H, W, 3+32*3]
    
    # Permute to NCHW for network
    net_input = net_input.permute([0, 3, 1, 2])  # [1, C, H, W]
    
    # Forward pass
    with torch.no_grad():
        out = model(net_input)
    
    # Decode MPI
    dep_var = {
        'ref_img': ref_view['image'].unsqueeze(0),  # needs batch dim
        'mpi_planes': psv_planes,
    }
    rgba_layers = mpi_from_net_output(out, dep_var)
    
    return rgba_layers, psv_planes


def render_view(rgba_layers, ref_pose, tgt_pose, mpi_planes, intrinsics, device='cuda'):
    """Render target view from MPI."""
    rel_pose = torch.matmul(tgt_pose, torch.inverse(ref_pose))
    rendered = mpi_render_view_torch(
        rgba_layers, 
        rel_pose.unsqueeze(0), 
        mpi_planes, 
        intrinsics.unsqueeze(0),
        device=device
    )
    return rendered[0]  # remove batch dim


def save_image_tensor(img_tensor, save_path):
    """Save image tensor [-1,1] HWC to file."""
    try:
        img_np = (((img_tensor[:, :, :3] + 1.0) / 2.0).clamp(0, 1) * 255.0).byte().cpu().numpy()
        Image.fromarray(img_np).save(save_path)
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")
        print(f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
        raise


def compute_psnr(img1, img2):
    """Compute PSNR between two tensors in [-1,1]."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


import struct
import numpy as np

def read_colmap_cameras_bin(path):
    """
    Reads COLMAP cameras.bin file.
    Returns a dict: {camera_id: {'model': str, 'width': int, 'height': int, 'params': np.array}}
    """
    cameras = {}
    CAMERA_MODEL_IDS = {
        0: 'SIMPLE_PINHOLE',
        1: 'PINHOLE',
        2: 'SIMPLE_RADIAL',
        3: 'RADIAL',
        4: 'OPENCV',
        5: 'OPENCV_FISHEYE',
        6: 'FULL_OPENCV',
        7: 'FOV',
        8: 'SIMPLE_RADIAL_FISHEYE',
        9: 'RADIAL_FISHEYE',
        10: 'THIN_PRISM_FISHEYE',
    }
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            model = CAMERA_MODEL_IDS.get(model_id, "UNKNOWN")
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            if model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE']:
                num_params = 3
            elif model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FOV', 'RADIAL_FISHEYE']:
                num_params = 4
            elif model in ['RADIAL']:
                num_params = 5
            elif model in ['FULL_OPENCV', 'THIN_PRISM_FISHEYE']:
                num_params = 8
            else:
                raise ValueError(f"Unknown camera model: {model}")
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': np.array(params)
            }
    return cameras

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def read_colmap_images_bin(path):
    """
    Reads COLMAP images.bin file.
    Returns a dict: {image_id: {'qvec': np.array, 'tvec': np.array, 'camera_id': int, 'name': str, 'Rt': np.array}}
    For each image, also computes the 3x4 [R|t] matrix (world-to-camera) flattened to shape (12,).
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(8*4))
            tvec = struct.unpack("<ddd", f.read(8*3))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            # skip 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * (8 + 8 + 4))  # x, y, point3D_id

            images[image_id] = {
                'qvec': np.array(qvec),
                'tvec': np.array(tvec),
                'camera_id': camera_id,
                'name': name,
            }

    return images

def get_camera_intrinsics(camera):
    """
    Given a camera dict from read_colmap_cameras_bin, return 3x3 intrinsics matrix.
    Here, we assume that we only have one shared camera for the scene.
    """
    try:
        camera = list(camera.values())[0]
    except:
        raise ValueError("Multiple cameras found in the scene. Requires one.")
    model = camera['model']
    params = camera['params']
    if model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE']:
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FOV', 'RADIAL_FISHEYE']:
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    elif model in ['RADIAL']:
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model in ['FULL_OPENCV', 'THIN_PRISM_FISHEYE']:
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    else:
        raise ValueError(f"Unknown camera model: {model}")
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    return K

def get_camera_extrinsics(image):
    """
    Given an image dict from read_colmap_images_bin, return 4x4 world-to-camera extrinsic matrix.
    """
    qvec = image['qvec']
    tvec = image['tvec']
    R = qvec2rotmat(qvec)
    t = np.array(tvec).reshape(3, 1)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:] = t
    return extrinsic

def get_flat_camera_params(extr, intr):
    """
    Convert extrinsic and intrinsic matrices to flat camera parameters expected by the model.
    Focal lengths, principal points, then row-major order of the 3x4 extrinsic matrix.
    [fx, fy, cx, cy, 0.0, 0.0, r1, r2, r3, tx, r4, r5, r6, ty, r7, r8, r9, tz]
    (0s kept for compatibility with RE10K.)
    """
    fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    tx, ty, tz = extr[0,3], extr[1,3], extr[2,3]
    r1, r2, r3 = extr[0,0], extr[1,0], extr[2,0]
    r4, r5, r6 = extr[0,1], extr[1,1], extr[2,1]
    r7, r8, r9 = extr[0,2], extr[1,2], extr[2,2]
    return np.array([fx, fy, cx, cy, 0.0, 0.0, r1, r2, r3, tx, r4, r5, r6, ty, r7, r8, r9, tz])


def load_custom_scene(colmap_path):
    """Load a scene from custom (COLMAP) format."""
    images = []
    images_path = os.path.join(colmap_path, 'images')
    for img_name in sorted(os.listdir(images_path)):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(images_path, img_name)
            img = Image.open(img_path)
            img = img.convert('RGB')
            images.append(np.array(img))

    cameras_bin = os.path.join(colmap_path, 'sparse', '0', 'cameras.bin') # intrinsics
    images_bin = os.path.join(colmap_path, 'sparse', '0', 'images.bin') # extrinsics
    intrinsics = get_camera_intrinsics(read_colmap_cameras_bin(cameras_bin))
    images_info = read_colmap_images_bin(images_bin)

    extrinsics = []
    for info in images_info.values():
        # Compute 3x4 Rt matrix (world-to-camera)
        R = qvec2rotmat(info['qvec'])
        t = np.array(info['tvec']).reshape(3, 1)
        Rt = np.concatenate([R, t], axis=1).astype(np.float64)  # shape (3,4)
        extrinsics.append(Rt)

    # normalize intrinsics via W,H of image
    H, W = images[0].shape[:2] # H,W,C
    intrinsics[0, :] = intrinsics[0, :] / W
    intrinsics[1, :] = intrinsics[1, :] / H

    cameras = [torch.tensor(get_flat_camera_params(extrinsics[i], intrinsics)) for i in range(len(extrinsics))]
    cameras = torch.stack(cameras, dim=0)
    return {'images': images, 'cameras': cameras, 'key': os.path.basename(colmap_path)}


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_planes = ckt.get('num_planes', args.num_planes)
    
    # Load model
    model = StereoMagnificationModel(num_mpi_planes=num_planes)
    model.load_state_dict(ckt['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load scene
    print(f"Loading scene {args.data_path}")
    if args.dataset_type == 're10k':
        scene = load_re10k_scene(args.data_path, args.scene_idx)
        all_images = scene['images']
        print(all_images)
        print(scene['cameras'])
        all_cameras = scene['cameras']
        scene_id = scene['key']
    elif args.dataset_type == 'colmap': # COLMAP format directory
        scene = load_custom_scene(args.data_path)
        all_images = scene['images']
        all_cameras = scene['cameras']
        scene_id = scene['key']
    else:
        raise NotImplementedError("Only RE10K and custom datasets currently supported")
    
    num_views = len(all_images)
    print(f"Scene has {num_views} views")
    
    # Determine image size
    if args.img_size is not None:
        img_size = tuple(args.img_size)
    else:
        # Use original size from first image
        from torchvision.io import decode_image
        if args.dataset_type == 're10k':
            # NOTE: decode_img gets it in (3, H, W)
            temp_img = decode_image(all_images[0], mode="RGB")
            _, H, W = temp_img.shape
            img_size = (W, H)
        else: 
            # originally, images are of (H, W, 3)
             H, W, _= all_images[0].shape
             img_size = (W, H)
    
    if args.dataset_type == 'colmap':  # get into compatible format
        # Convert all images to torch tensors in (3, H, W) format and update all_images in-place
        all_images = [torch.tensor(img).permute(2, 0, 1).contiguous() for img in all_images]
    print(f"Using image size: {img_size}")
    
    # Prepare reference and source views
    print(f"Using ref_idx={args.ref_idx}, src_idx={args.src_idx}")
    ref_view = prepare_view(all_images[args.ref_idx], all_cameras[args.ref_idx], 
                             img_size, device=device, decode=False if args.dataset_type == 'colmap' else True)
    src_view = prepare_view(all_images[args.src_idx], all_cameras[args.src_idx], 
                             img_size, device=device, decode=False if args.dataset_type == 'colmap' else True)
    
    # Build MPI
    print("Building MPI...")
    rgba_layers, mpi_planes = build_mpi(model, ref_view, src_view, num_planes, device=device)
    
    # Save MPI layers if requested
    if args.save_mpi_layers:
        print("Saving MPI layer visualizations...")
        mpi_grid = visualize_mpi_layers(rgba_layers, mpi_planes, max_cols=8, show_alpha=True)
        Image.fromarray(mpi_grid).save(os.path.join(args.output_dir, 'mpi_layers.png'))
    
    # Render all views
    print(f"Rendering {num_views} views...")
    psnrs = []
    
    for i in tqdm(range(num_views)):
        try:
            # Prepare target view
            tgt_view = prepare_view(all_images[i], all_cameras[i], img_size, device=device, decode=False if args.dataset_type == 'colmap' else True)
            
            # Render
            rendered = render_view(
                rgba_layers, 
                ref_view['pose'], 
                tgt_view['pose'], 
                mpi_planes, 
                tgt_view['intrinsics'],
                device=device
            )
            
            # Save
            os.makedirs(os.path.join(args.output_dir, 'renders'), exist_ok=True)
            save_path = os.path.join(args.output_dir, 'renders', f'render_{i:04d}.png')
            save_image_tensor(rendered, save_path)
            
            # Also save ground truth for comparison
            os.makedirs(os.path.join(args.output_dir, 'gt'), exist_ok=True)
            gt_path = os.path.join(args.output_dir, 'gt', f'gt_{i:04d}.png')
            save_image_tensor(tgt_view['image'], gt_path)
            
            # Compute metrics if requested
            if args.compute_metrics:
                psnr = compute_psnr(rendered[:, :, :3], tgt_view['image'][:, :, :3])
                psnrs.append(psnr.item())
        except Exception as e:
            print(f"Error processing view {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Report metrics
    if args.compute_metrics:
        print(f"\nAverage PSNR: {np.mean(psnrs):.2f} dB")
        print(f"PSNR std: {np.std(psnrs):.2f} dB")
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Scene: {scene_id if args.dataset_type == 're10k' else args.scene_idx}\n")
            f.write(f"Ref idx: {args.ref_idx}, Src idx: {args.src_idx}\n")
            f.write(f"Average PSNR: {np.mean(psnrs):.2f} dB\n")
            f.write(f"PSNR std: {np.std(psnrs):.2f} dB\n")
            f.write(f"\nPer-view PSNR:\n")
            for i, psnr in enumerate(psnrs):
                f.write(f"View {i:04d}: {psnr:.2f} dB\n")
    
    print(f"\nOutputs saved to {args.output_dir}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

