# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation and https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    avg_bce_loss = AverageMeter()  # Add this for BCE loss tracking

    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader):
        # For face manipulation data, we expect images and masks
        images, masks = batch['image'], batch['mask']  # Modified to get mask instead of labels
        images = images.cuda()
        masks = masks.float().cuda()  # Ensure masks are float for BCE loss
        
        # Forward pass to get probability outputs
        outputs = model(images)
        
        # Apply sigmoid to get probabilities in [0,1] range
        outputs = torch.sigmoid(outputs)
        
        # Calculate BCE loss
        loss = F.binary_cross_entropy(outputs, masks, reduction='mean')
        
        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # Update average loss
        avg_bce_loss.update(reduced_loss.item())

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer,
                              base_lr,
                              num_iters,
                              i_iter+cur_iters)
                              
        # Print progress
        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, BCE Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], 
                      avg_bce_loss.average())
            logging.info(msg)

    # Log to tensorboard
    writer.add_scalar('train_bce_loss', avg_bce_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            images, masks = batch['image'], batch['mask']
            images = images.cuda()
            masks = masks.float().cuda()  # Ensure mask is float for BCE loss
            
            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            # Calculate BCE loss
            loss = F.binary_cross_entropy(outputs, masks, reduction='mean')
            
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
                
            ave_loss.update(reduced_loss.item())

    # Log to tensorboard
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_bce_loss', ave_loss.average(), global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    return ave_loss.average()

def train_subprocess(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    avg_bce_loss = AverageMeter()  # Add this for BCE loss tracking

    tic = time.time()
    cur_iters = epoch*epoch_iters

    for i_iter, batch in enumerate(trainloader):
        # For face manipulation data, we expect images and masks
        images, masks = batch['image'], batch['mask']  # Modified to get mask instead of labels
        images = images.cuda()
        masks = masks.float().cuda()  # Ensure masks are float for BCE loss
        
        # Forward pass to get probability outputs
        outputs = model(images)
        
        # Apply sigmoid to get probabilities in [0,1] range
        outputs = torch.sigmoid(outputs)
        
        # Calculate BCE loss
        loss = F.binary_cross_entropy(outputs, masks, reduction='mean')
        
        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # Update average loss
        avg_bce_loss.update(reduced_loss.item())

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer,
                              base_lr,
                              num_iters,
                              i_iter+cur_iters)
                              
        # Print progress
        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, BCE Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], 
                      avg_bce_loss.average())
            logging.info(msg)


def validate_subprocess(config, valloader, model):
    model.eval()
    ave_loss = AverageMeter()
    
    with torch.no_grad():
        for idx, batch in enumerate(valloader):
            images, masks = batch['image'], batch['mask']
            images = images.cuda()
            masks = masks.float().cuda()  # Ensure mask is float for BCE loss
            
            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            # Calculate BCE loss
            loss = F.binary_cross_entropy(outputs, masks, reduction='mean')
            
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
                
            ave_loss.update(reduced_loss.item())
    
    return ave_loss.average()

def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
                logging.info(IoU_array)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
