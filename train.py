import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm

from dataset import RealEstateDataset
from networks import StereoMagnificationModel, VGGPerceptualLoss
from utils import *

img_size = (640, 360)  # W,H
num_planes = 32
lr = 2e-4
batch_size = 1
end_epoch = 20
checkpoint = None
print_freq = 20
data_dir = "/workspace/re10kvol/re10k"
save_dir = 'checkpoints'

parser = ArgumentParser(description="Train for MPIs")

parser.add_argument('--save_dir', default=save_dir, type=str, help="Directory of save checkpoint")
parser.add_argument('--data_dir', default=data_dir, type=str, help="Directory of real-estate data")
parser.add_argument('--checkpoint', default=checkpoint, type=str, help="Directory of load checkpoint")
parser.add_argument('--img_size', default=img_size, type=int, help="training data image resolution")
parser.add_argument('--num_planes', default=num_planes, type=int, help="MPIs depths")
parser.add_argument('--end_epoch', default=end_epoch, type=int, help="Training epoch size")
parser.add_argument('--lr', default=lr, type=float, help="Start learning rate")
parser.add_argument('--batch_size', default=batch_size, type=int, help="Mini batch size")
parser.add_argument('--print_freq', default=print_freq, type=int, help="Training loss print frequency")
parser.add_argument('--seed', default=7, type=int, help="Training seed")


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
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    logger = get_logger()

    for epoch in range(start_epoch, args.end_epoch):
        train_loss = train(train_loader=train_loader,
                          model=model,
                          optimizer=optimizer,
                          epoch=epoch,
                          logger=logger)
        writer.add_scalar('Train_Loss', train_loss, epoch)
        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                          model=model,
                          logger=logger)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model.state_dict(), optimizer, best_loss, is_best, args.save_dir, args.num_planes)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    print("Training")
    losses = AverageMeter()
    criterion = VGGPerceptualLoss().to(device)
    for i, (img, dep) in enumerate(tqdm(train_loader), desc="Training MPI"):
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print status
        if i % print_freq == 0:
            status = 'Epoch: [{0}][step: {1}]\t' \
                    'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'.format(epoch, i, loss=losses)
            logger.info(status)
    return losses.avg


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()
    l2_loss = nn.MSELoss().to(device)

    for img, dep in tqdm(valid_loader):
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

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\n'.format(loss=losses)
    logger.info(status)
    return losses.avg


def save_checkpoint(epoch, epochs_since_improvement, state_dict, optimizer, loss, is_best, dir, num_planes):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'state_dict': state_dict,
             'optimizer': optimizer,
             'num_planes': num_planes}

    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, 'checkpoint.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(dir, 'BEST_checkpoint.tar'))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    train_net(args)
