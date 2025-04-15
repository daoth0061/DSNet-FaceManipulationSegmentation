# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation and https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import torch.optim as optim
import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate, train_subprocess, validate_subprocess
from utils.utils import create_logger, FullModel
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/kaggle/working/DSNet-FaceManipulationSegmentation/configs/FaceManipulationDetection/AttGAN/ds_base_attgan.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    # parser.add_argument("--local_rank", type=int, default=-1)       

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():

    #Fix seed
    args = parse_args()
    # torch.autograd.set_detect_anomaly(True)
    print(f"I am process {args.local_rank}.")
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    #Set up log
    if args.local_rank <= 0:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'dsnet_m')

        logger.info(pprint.pformat(args))
        logger.info(config)
        print(tb_log_dir)
        writer_dict = {
            'writer': SummaryWriter(tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Set up GPUs
    gpus = list(config.GPUS)


    print(gpus)
    if torch.cuda.device_count() != len(gpus):
        print(len(gpus))
        print(torch.cuda.device_count())
        print("The gpu numbers do not match!")


    distributed = args.local_rank >= 0
    if distributed:
        print("---------------devices:", args.local_rank)
        device = torch.device('cuda:{}'.format(args.local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )  
        # return 0

    
    if distributed and args.local_rank == 0:
        print(final_output_dir)
        # this_dir = os.path.dirname(os.getcwd())
        this_dir = '/kaggle/working/DSNet-FaceManipulationSegmentation/tools'
        print(f"this_dir: {this_dir}")
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../models'), models_dst_dir)
        print(os.path.join(this_dir, '../models'))

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)   

    # batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # Define paths
    fake_dir = '/kaggle/input/dataset-attrgan/fake_attrGAN/fake_attrGAN'
    real_dir = '/kaggle/input/dataset-attrgan/real-20250326T031740Z-001/real'
    mask_dir = '/kaggle/input/masked-dataset-newversion/mask'
    high_quality_images_path = '/kaggle/input/attgan-filtering/evaluation_results/images_threshold_0.95.txt'
    
    # Create datasets
    full_dataset = eval('datasets.'+config.DATASET.DATASET)(
        fake_dir=fake_dir,
        real_dir=real_dir,
        mask_dir=mask_dir,
        high_quality_images_path=high_quality_images_path,
        split='train'
    )

    # Split into train/val/test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Get indices for the split
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Save test indices to a file
    if args.local_rank <= 0:
        with open('test_indices.txt', 'w') as f:
            for idx in test_indices:
                f.write(f"{idx}\n")

    # Create subsets using the indices
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    # test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_sampler = get_sampler(train_dataset)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler = train_sampler)
    
    val_sampler = get_sampler(val_dataset)

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=val_sampler)

    # test_sampler = get_sampler(test_dataset)

    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=config.TEST.BATCH_SIZE_PER_GPU,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True,
    #     sampler=test_sampler)
    


    # Create model
    model = models.dsnet.get_seg_model(config, imgnet_pretrained=True)

    if distributed:
        model = model.to(device)

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
    )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # # optimizer
    if config.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                           lr=config.TRAIN.LR,
                           weight_decay=config.TRAIN.WD)
    else:
        raise ValueError('Only Support Adam optimizer')



    epoch_iters = np.int64(len(train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    
    best_loss = float('inf')  # Track best loss instead of mIoU
    last_epoch = config.TRAIN.BEGIN_EPOCH
    valid_loss = 0
    flag_rm = config.TRAIN.RESUME

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(config.MODEL.PRETRAINED)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_loss = checkpoint.get('best_loss', float('inf'))  # Get best_loss instead of best_mIoU
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.local_rank <= 0:
                logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = end_epoch
    base_lr = config.TRAIN.LR

    for epoch in range(last_epoch, real_end):
        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)
        if args.local_rank <= 0:
            # Call existing train function, assuming it now properly calculates BCE loss
            train(config, epoch, config.TRAIN.END_EPOCH, 
                    epoch_iters, base_lr, num_iters,
                    trainloader, optimizer, model, writer_dict)
        else:
            # Call subprocess for distributed training
            train_subprocess(config, epoch, config.TRAIN.END_EPOCH, 
                    epoch_iters, base_lr, num_iters,
                    trainloader, optimizer, model)

        # Validation check at specified intervals
        if flag_rm == 1 or (epoch % 2 == 0 and epoch <= 50) or (epoch % 20 == 0 and epoch > 50 and epoch <= 180) or (epoch > 180 and epoch % 2 == 0) or (epoch > 235): 
            # Modify validate function to return only BCE loss, not mIoU
            if args.local_rank <= 0:
                valid_loss = validate(config, valloader, model, writer_dict)
            else:
                # Call subprocess for distributed validation
                valid_loss = validate_subprocess(config, valloader, model)

        if flag_rm == 1:
            flag_rm = 0
        
        if args.local_rank <= 0:
        
            # Save best model based on loss instead of mIoU
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dsnet_face.pth'))
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dsnet_face.pt'))
                
                # # Save additional checkpoints for particularly good models
                # if best_loss < 0.10:  # Example threshold, adjust as needed
                #     torch.save(model.module.state_dict(),
                #         os.path.join(final_output_dir, 'best_dsnet_face_{:.6f}.pth'.format(best_loss)))

                #     torch.save({
                #     'epoch': epoch+1,
                #     'best_loss': best_loss,
                #     'state_dict': model.module.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     }, os.path.join(final_output_dir,'best_dsnet_face_{:.6f}.pth.tar'.format(best_loss)))
                # else:
                #     torch.save({
                #     'epoch': epoch+1,
                #     'best_loss': best_loss,
                #     'state_dict': model.module.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     }, os.path.join(final_output_dir,'best_dsnet_face.pth.tar'))
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint_dsnet_face.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_loss': best_loss,  # Save best_loss instead of best_mIoU
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint_dsnet_face.pth.tar'))
            
            # Log BCE loss instead of mIoU
            msg = 'Loss: {:.3f}, Valid Loss: {:.4f}, Best Loss: {:.4f}'.format(
                        0.0, valid_loss, best_loss)  # Train loss placeholder
            logging.info(msg)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict(),
                os.path.join(final_output_dir, 'final_dsnet_face.pt'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % int((end-start)/3600))
        logger.info('Done')
if __name__ == '__main__':
    main()

   