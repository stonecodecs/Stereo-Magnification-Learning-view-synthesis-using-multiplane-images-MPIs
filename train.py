import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
import wandb

from dataset import RealEstateDataset
from networks import StereoMagnificationModel, VGGPerceptualLoss
from utils import *

img_size = (360, 640)  # W,H
num_planes = 32
lr = 2e-4
batch_size = 1
end_epoch = 20
checkpoint = None
print_freq = 100
data_dir = "/workspace/re10kvol/re10k"
save_dir = 'checkpoints'
seed = 7

parser = ArgumentParser(description="Train for MPIs")

parser.add_argument('--save_dir', default=save_dir, type=str, help="Directory of save checkpoint")
parser.add_argument('--data_dir', default=data_dir, type=str, help="Directory of real-estate data")
parser.add_argument('--checkpoint', default=checkpoint, type=str, help="Directory of load checkpoint to resume training.")
parser.add_argument('--img_size', default=img_size, type=int, help="training data image resolution")
parser.add_argument('--num_planes', default=num_planes, type=int, help="MPIs depths")
parser.add_argument('--end_epoch', default=end_epoch, type=int, help="Training epoch size")
parser.add_argument('--lr', default=lr, type=float, help="Start learning rate")
parser.add_argument('--batch_size', default=batch_size, type=int, help="Mini batch size")
parser.add_argument('--print_freq', default=print_freq, type=int, help="Training loss print frequency")
parser.add_argument('--seed', default=seed, type=int, help="Training seed")
parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
parser.add_argument('--log_img_every', default=1000, type=int, help="Log images every N training iterations")
parser.add_argument('--save_every_n_iterations', default=None, type=int, help="Saves a checkpoint every N training iterations.")


def train_net(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # load checkpoint
    if args.checkpoint is not None:
        ckt = torch.load(args.checkpoint)
        epoch = ckt['epoch']
        epochs_since_improvement = ckt['epochs_since_improvement']
        best_loss = ckt['loss']
        model = StereoMagnificationModel(num_mpi_planes=ckt['num_planes'])
        model.load_state_dict = ckt['state_dict']
        optimizer = ckt['optimizer']
    else:
        model = StereoMagnificationModel(num_mpi_planes=args.num_planes)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = RealEstateDataset(args.data_dir, img_size=args.img_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    valid_dataset = RealEstateDataset(args.data_dir, img_size=args.img_size, is_valid=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    logger = get_logger()

    for epoch in range(start_epoch, args.end_epoch):
        train_loss = train(train_loader=train_loader,
                          model=model,
                          optimizer=optimizer,
                          epoch=epoch,
                          logger=logger,
                          log_img_every=args.log_img_every,
                          use_wandb=args.wandb,
                          save_every_n_iterations=args.save_every_n_iterations)
        writer.add_scalar('Train_Loss', train_loss, epoch)
        if args.wandb:
            wandb.log({'train/loss_epoch': train_loss, 'epoch': epoch})
        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                          model=model,
                          logger=logger,
                          epoch=epoch,
                          log_img_every=args.log_img_every,
                          use_wandb=args.wandb,
                          save_every_n_iterations=None) # no checkpoint saving for validation

        writer.add_scalar('Valid_Loss', valid_loss, epoch)
        if args.wandb:
            wandb.log({'valid/loss_epoch': valid_loss, 'epoch': epoch})

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model.state_dict(), optimizer, best_loss, is_best, args.save_dir, args.num_planes, train_iteration=None)


def train(train_loader, model, optimizer, epoch, logger, log_img_every=200, use_wandb=False, save_every_n_iterations=None):
    model.train()  # train mode (dropout and batchnorm is used)

    print("Training")
    losses = AverageMeter()
    criterion = VGGPerceptualLoss().to(device)
    iteration = 0
    for i, (img, dep) in enumerate(tqdm(train_loader, desc="Training MPI")):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device, non_blocking=True)
        for k, v in dep.items():
            if isinstance(v, torch.Tensor): # only tensors
                dep[k] = v.type(torch.FloatTensor).to(device, non_blocking=True)

        # Forward prop.
        out = model(img)  # [N, 3, 320, 320]

        # Calculate loss
        loss = criterion(out, dep)
        losses.update(loss.item())

        # wandb: per-iter loss
        if use_wandb:
            wandb.log({'train/loss': loss.item(), 'epoch': epoch, 'iter': i})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print status
        if i % print_freq == 0:
            status = 'Epoch: [{0}][step: {1}]\t' \
                    'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'.format(epoch, i, loss=losses)
            logger.info(status)

        # Periodic image logging (first item in batch)
        if use_wandb and (i % log_img_every == 0):
            try:
                with torch.no_grad():
                    rgba_layers = mpi_from_net_output(out, dep)
                    rel_pose = torch.matmul(dep['tgt_img_cfw'], dep['ref_img_wfc'])
                    pred_image = mpi_render_view_torch(rgba_layers, rel_pose, dep['mpi_planes'][0], dep['intrinsics'])
                    # take first sample and convert to uint8 HWC
                    pred_vis = (((pred_image[0, :, :, :3] + 1.0) / 2.0).clamp(0,1) * 255.0).byte().cpu().numpy()
                    tgt_vis = (((dep['tgt_img'][0, :, :, :3] + 1.0) / 2.0).clamp(0,1) * 255.0).byte().cpu().numpy()
                    ref_vis = (((dep['ref_img'][0, :, :, :3] + 1.0) / 2.0).clamp(0,1) * 255.0).byte().cpu().numpy()
                    wandb.log({
                        'train/pred_image': wandb.Image(pred_vis),
                        'train/target_image': wandb.Image(tgt_vis),
                        'train/ref_image': wandb.Image(ref_vis),
                        'epoch': epoch,
                        'iter': i
                    })
            except Exception:
                pass

        # save model every N training iterations
        # NOTE:
        if save_every_n_iterations is not None and iteration != 0 and iteration % save_every_n_iterations == 0:
            save_checkpoint(epoch, iteration, model.state_dict(), optimizer, float("inf"), False, args.save_dir, args.num_planes, train_iteration=iteration)

        iteration += 1
    return losses.avg


def valid(valid_loader, model, logger, epoch=0, log_img_every=200, use_wandb=False):
    model.eval()

    losses = AverageMeter()
    l2_loss = nn.MSELoss().to(device)

    for i, (img, dep) in enumerate(tqdm(valid_loader, desc="Validating MPI")):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device, non_blocking=True)
        for k, v in dep.items():
            if isinstance(v, torch.Tensor): # only tensors
                dep[k] = v.type(torch.FloatTensor).to(device, non_blocking=True)
        target = dep['tgt_img'].to(device)

        # Forward prop.
        out = model(img)
        rgba_layers = mpi_from_net_output(out, dep)
        rel_pose = torch.matmul(dep['tgt_img_cfw'], dep['ref_img_wfc']).to(device)
        output_image = mpi_render_view_torch(rgba_layers, rel_pose, dep['mpi_planes'][0], dep['intrinsics']).to(device)

        # Calculate loss
        loss = l2_loss(output_image, target)
        losses.update(loss.item())

        # Periodic image logging
        if use_wandb and (i % log_img_every == 0):
            try:
                pred_vis = (((output_image[0, :, :, :3] + 1.0) / 2.0).clamp(0,1) * 255.0).byte().cpu().numpy()
                tgt_vis = (((target[0, :, :, :3] + 1.0) / 2.0).clamp(0,1) * 255.0).byte().cpu().numpy()
                ref_vis = (((dep['ref_img'][0, :, :, :3] + 1.0) / 2.0).clamp(0,1) * 255.0).byte().cpu().numpy()
                wandb.log({
                    'valid/pred_image': wandb.Image(pred_vis),
                    'valid/target_image': wandb.Image(tgt_vis),
                    'valid/ref_image': wandb.Image(ref_vis),
                    'epoch': epoch,
                    'valid/iter': i
                })
            except Exception:
                pass

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\n'.format(loss=losses)
    logger.info(status)
    return losses.avg


def save_checkpoint(epoch, epochs_since_improvement, state_dict, optimizer, loss, is_best, dir, num_planes, train_iteration=None):
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'loss': loss,
        'state_dict': state_dict,
        'optimizer': optimizer,
        'num_planes': num_planes
    }

    if not os.path.exists(dir):
        os.makedirs(dir)


    if train_iteration is not None: # if save_every_n_iterations is not None, save the checkpoint with the current iteration number
        filename = os.path.join(dir, f'checkpoint_{epoch}_{train_iteration}.tar') # current checkpoint
    else:
        filename = os.path.join(dir, 'checkpoint.tar') # current end-of-epoch checkpoint
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(dir, 'BEST_checkpoint.tar'))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    # wandb config
    if args.wandb:
        project_name = "mpi-re10k"
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "end_epoch": args.end_epoch,
            "img_size": args.img_size,
            "num_planes": args.num_planes,
            "log_img_every": args.log_img_every,
        }
        try:
            wandb.init(project=project_name, config=config)
        except Exception:
            pass
    train_net(args)
