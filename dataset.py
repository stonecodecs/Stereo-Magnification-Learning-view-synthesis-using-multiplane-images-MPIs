import os
import PIL
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import decode_image

from utils import *

def flatten_list(l):
  # flatten nested lists to 1D
  if isinstance(l, list):
    return [item for sublist in l for item in flatten_list(sublist)]
  else:
    return [l]

class RealEstateDataset(torch.utils.data.IterableDataset):
  # for pixelsplat preprocessed version of RE10K
  def __init__(self, dataset_path, is_valid=False, min_dist=16e3, max_dist=500e3, img_size=(512, 288), num_planes=32):
    self.is_valid = is_valid
    self.dataset_path = dataset_path
    self.min_dist = min_dist
    self.max_dist = max_dist
    self.num_planes = num_planes
    if isinstance(img_size, int):
      self.img_size = (img_size, img_size)
    else:
      self.img_size = img_size

    metadataBasePath = os.path.join(dataset_path, "test" if is_valid else "train")
    self.shards = [os.path.join(metadataBasePath, shard_id) for shard_id in os.listdir(metadataBasePath)]
    self.current_shard_idx = 0

  def new_empty(self):
    return []
  
  def _draw(self, scene):
    img_range = range(len(scene['timestamps']))
    ref_img_idx = np.random.choice(img_range)
    base_timestamp = scene['timestamps'][ref_img_idx]
    near_range = list(filter(lambda i: abs(base_timestamp - scene['timestamps'][i]) >= self.min_dist and  abs(base_timestamp - scene['timestamps'][i]) <= self.max_dist, img_range))    
    assert(len(near_range) >= 2)

    src_img_idx = np.random.choice(near_range)
    tgt_img_idx = np.random.choice([i for i in near_range if i != src_img_idx])
    return [ref_img_idx, src_img_idx, tgt_img_idx]

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # Single-process data loading
        start_idx = 0
        end_idx = len(self.shards)
    else:  # Multi-process data loading
        # Split shards among workers
        per_worker = int(np.ceil(len(self.shards) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        start_idx = worker_id * per_worker
        end_idx = min(start_idx + per_worker, len(self.shards))
    
    # Iterate over the assigned shards (each shard has multiple scenes)
    for i in range(start_idx, end_idx):
        shard_path = self.shards[i]
        scenes_in_shard = torch.load(shard_path)
        
        # Now iterate over each scene
        for scene in scenes_in_shard:
          # Dict scene:
          # url(str), timestamps (M,), cameras (M,18), images M[enc. jpeg], key (hash str)
            if self.is_valid: # tests the first 3 images
              indexes = [0,1,2]
            else: # during training, draw random samples
              indexes = self._draw(scene)
            
            all_images = scene['images']
            all_cameras = scene['cameras']
            all_timestamps = scene['timestamps']

            # selected scene data
            metadata = []

            # for selected index, get camera params
            for index in indexes:
              # extract camera parameters
              fx, fy, cx, cy = all_cameras[index][0:4]
              Rt_flat = all_cameras[index][6:]
              # expand to pixels rather than normalized intrinsics
              intr = make_intrinsics_matrix(
                self.img_size[0] * fx,
                self.img_size[1] * fy,
                self.img_size[0] * cx,
                self.img_size[1] * cy,
              )
              extr = to_homogenous(make_extrinsics_matrix(Rt_flat))

              # extract images
              img = decode_image(all_images[index], mode="RGB") # (C,H,W)
              scaled_img = F.interpolate(img.unsqueeze(0), self.img_size).squeeze(0)
              img_tensor = preprocess_image_torch(scaled_img/255.0).permute(1, 2, 0) # [-1,1] HWC

              metadata.append({
                'timestamp': all_timestamps[index],
                'intrinsics': intr,
                'pose': extr,
                'image': img_tensor
              })
            
            ref_img = metadata[0] # anchor view
            src_img = metadata[1] # support view
            tgt_img = metadata[2] # target to learn
            
            # uniform disparity sampling
            psv_planes = torch.Tensor(inv_depths(1, 100, self.num_planes))
            curr_pose = torch.matmul(src_img['pose'], torch.inverse(ref_img['pose']))
            curr_psv = plane_sweep_torch_one(src_img['image'], psv_planes, curr_pose, src_img['intrinsics'])
            
            # N,H,W,3(D+1)
            net_input = torch.cat([torch.unsqueeze(ref_img['image'], 0), curr_psv], 3)
            dep_var = {
                'tgt_img_cfw': tgt_img['pose'],
                'tgt_img': tgt_img['image'],
                'ref_img': ref_img['image'],
                'ref_img_wfc': torch.inverse(ref_img['pose']),
                'intrinsics': src_img['intrinsics'],
                'mpi_planes': psv_planes,
                'data_id': scene['key'] # replacing text_id
            }
            yield [torch.squeeze(net_input).permute([2, 0, 1]), dep_var]

if __name__ == '__main__':
  train_data = RealEstateDataset("/workspace/re10kvol/re10k")
  # A DataLoader with multiple workers will load shards in parallel
  train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=2, num_workers=8, pin_memory=True)

  for i, (input_data, target_data) in enumerate(train_loader):
      print(f"Batch {i+1} loaded.")
      if i > 5: # Just to test and stop
        break