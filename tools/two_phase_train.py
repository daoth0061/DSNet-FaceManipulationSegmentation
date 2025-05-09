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
# from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
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


# Split datasets for both fake and real datasets with the same ratio
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    """Split a dataset into train, validation, and test sets with specified ratios."""
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Get indices for the split
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

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
    
    # Create fake datasets
    full_dataset_fake = eval('datasets.'+config.DATASET.DATASET)(
        fake_dir=fake_dir,
        real_dir=None,
        mask_dir=mask_dir,
        high_quality_images_path=high_quality_images_path,
        split='train'
    )

    # Create real datasets
    full_dataset_real = eval('datasets.'+config.DATASET.DATASET)(
        fake_dir=None,
        real_dir=real_dir,
        mask_dir=None,
        high_quality_images_path=None,
        split='train'
    )

    # Split both datasets with same ratio
    fake_train_indices, fake_val_indices, fake_test_indices = split_dataset(full_dataset_fake)
    real_train_indices, real_val_indices, real_test_indices = split_dataset(full_dataset_real)

    # Create subsets for fake dataset
    fake_train_dataset = torch.utils.data.Subset(full_dataset_fake, fake_train_indices)
    fake_val_dataset = torch.utils.data.Subset(full_dataset_fake, fake_val_indices)

    # Create subsets for real dataset
    real_train_dataset = torch.utils.data.Subset(full_dataset_real, real_train_indices)
    real_val_dataset = torch.utils.data.Subset(full_dataset_real, real_val_indices)

    # Save test indices to files
    if args.local_rank <= 0:
        # Save fake test indices
        with open('fake_test_indices.txt', 'w') as f:
            for idx in fake_test_indices:
                f.write(f"{idx}\n")
        
        # Save real test indices
        with open('real_test_indices.txt', 'w') as f:
            for idx in real_test_indices:
                f.write(f"{idx}\n")

    # Create combined validation and test sets for phase 2
    combined_val_dataset = torch.utils.data.ConcatDataset([fake_val_dataset, real_val_dataset])


    # Create data loaders for Phase 1 (real only)
    real_train_sampler = get_sampler(real_train_dataset)
    real_val_sampler = get_sampler(real_val_dataset)

    real_trainloader = torch.utils.data.DataLoader(
        real_train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and real_train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=real_train_sampler
    )

    real_valloader = torch.utils.data.DataLoader(
        real_val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=real_val_sampler
    )

    # Create data loaders for Phase 2 (fake for training, combined for validation)
    fake_train_sampler = get_sampler(fake_train_dataset)
    combined_val_sampler = get_sampler(combined_val_dataset)

    fake_trainloader = torch.utils.data.DataLoader(
        fake_train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and fake_train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=fake_train_sampler
    )

    combined_valloader = torch.utils.data.DataLoader(
        combined_val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=combined_val_sampler
    )
    


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

    
    best_loss = float('inf')  # Track best L1 loss instead of mIoU
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

    start_epoch = last_epoch
    end_epoch = config.TRAIN.END_EPOCH
    base_lr = config.TRAIN.LR

    # Calculate number of epochs for each phase
    total_epochs = end_epoch - start_epoch
    phase1_epochs = 5  
    phase2_epochs = total_epochs - phase1_epochs  # 85% for phase 2

    if args.local_rank <= 0:
        logger.info(f"Phase 1 (Real only): {phase1_epochs} epochs")
        logger.info(f"Phase 2 (Fake only): {phase2_epochs} epochs")

    # Calculate iterations per epoch for both phases
    real_epoch_iters = np.int64(len(real_train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    fake_epoch_iters = np.int64(len(fake_train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    if args.local_rank <= 0:
        start = timeit.default_timer()
        logger.info('Start training...')

    # Phase 1: Train on real images only
    for epoch in range(start_epoch, phase1_epochs):
        if args.local_rank <= 0:
            logger.info(f"Phase 1 - Epoch {epoch+1}/{phase1_epochs} (Real images only)")
        
        # Set appropriate sampler epoch
        if real_trainloader.sampler is not None and hasattr(real_trainloader.sampler, 'set_epoch'):
            real_trainloader.sampler.set_epoch(epoch)
        
        # Train with real images only
        if args.local_rank <= 0:
            train(config, epoch, phase1_epochs, 
                real_epoch_iters, base_lr, phase1_epochs * real_epoch_iters,
                real_trainloader, optimizer, model, writer_dict)
        else:
            train_subprocess(config, epoch, phase1_epochs, 
                    real_epoch_iters, base_lr, phase1_epochs * real_epoch_iters,
                    real_trainloader, optimizer, model)

        # Validation check at specified intervals or on the last epoch of phase 1
        if flag_rm == 1 or (epoch % 2 == 0) or (epoch == phase1_epochs - 1): 
            if args.local_rank <= 0:
                valid_loss = validate(config, real_valloader, model, writer_dict)
            else:
                valid_loss = validate_subprocess(config, real_valloader, model)

        if flag_rm == 1:
            flag_rm = 0
        
        if args.local_rank <= 0:
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best_dsnet_face_phase1.pth'))
                torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best_dsnet_face_phase1.pt'))
                
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint_dsnet_face_phase1.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'phase': 1,
                'best_loss': best_loss,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint_dsnet_face_phase1.pth.tar'))
            
            msg = 'Phase 1 - Loss: {:.3f}, Valid Loss: {:.4f}, Best Loss: {:.4f}'.format(
                        0.0, valid_loss, best_loss)
            logging.info(msg)
    

    # Phase 2: Train on fake images only, validate on combined
    phase2_best_loss = float('inf')
    start_epoch = max(last_epoch, phase1_epochs)

    for epoch in range(start_epoch, total_epochs):
        relative_epoch = epoch - start_epoch  # Relative epoch for phase 2
        
        if args.local_rank <= 0:
            logger.info(f"Phase 2 - Epoch {relative_epoch+1}/{phase2_epochs} (Fake images)")
        
        # Set appropriate sampler epoch
        if fake_trainloader.sampler is not None and hasattr(fake_trainloader.sampler, 'set_epoch'):
            fake_trainloader.sampler.set_epoch(relative_epoch)
        
        # Train with fake images only
        if args.local_rank <= 0:
            train(config, relative_epoch, phase2_epochs, 
                fake_epoch_iters, base_lr, phase2_epochs * fake_epoch_iters,
                fake_trainloader, optimizer, model, writer_dict)
        else:
            train_subprocess(config, relative_epoch, phase2_epochs, 
                    fake_epoch_iters, base_lr, phase2_epochs * fake_epoch_iters,
                    fake_trainloader, optimizer, model)
        
        # Validation check at specified intervals
        if (relative_epoch % 2 == 0 and relative_epoch <= 50) or \
        (relative_epoch % 20 == 0 and relative_epoch > 50 and relative_epoch <= 180) or \
        (relative_epoch > 180 and relative_epoch % 2 == 0) or \
        (relative_epoch > 235) or (relative_epoch == phase2_epochs - 1): 
            if args.local_rank <= 0:
                valid_loss = validate(config, combined_valloader, model, writer_dict)
            else:
                valid_loss = validate_subprocess(config, combined_valloader, model)
        
        # Save checkpoint
        if args.local_rank <= 0:
            if valid_loss < phase2_best_loss:
                phase2_best_loss = valid_loss
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dsnet_face_phase2.pth'))
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dsnet_face_phase2.pt'))
                    
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint_dsnet_face_phase2.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'phase': 2,
                'best_loss': phase2_best_loss,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint_dsnet_face_phase2.pth.tar'))
            
            msg = 'Phase 2 - Loss: {:.3f}, Valid Loss: {:.4f}, Best Loss: {:.4f}'.format(
                        0.0, valid_loss, phase2_best_loss)
            logging.info(msg)

    if args.local_rank <= 0:
        # Save final model
        torch.save(model.module.state_dict(),
                os.path.join(final_output_dir, 'final_dsnet_face.pt'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % int((end-start)/3600))
        logger.info('Done')
if __name__ == '__main__':
    main()

   