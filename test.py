#  best_mIoU = 0
# mean_IoU = 0
# last_epoch = config.TRAIN.BEGIN_EPOCH
# valid_loss = 0
# flag_rm = config.TRAIN.RESUME

# if config.TRAIN.RESUME:
#     model_state_file = os.path.join(config.MODEL.PRETRAINED)
#     if os.path.isfile(model_state_file):
#         checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
#         best_mIoU = checkpoint['best_mIoU']
#         last_epoch = checkpoint['epoch']
#         dct = checkpoint['state_dict']
        
#         model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
#     if distributed:
#         torch.distributed.barrier()

# # last_epoch = 0
# start = timeit.default_timer()
# end_epoch = config.TRAIN.END_EPOCH
# num_iters = config.TRAIN.END_EPOCH * epoch_iters
# # real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
# real_end = end_epoch
# base_lr = config.TRAIN.LR
# for epoch in range(last_epoch, real_end):

#     current_trainloader = trainloader
#     if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
#         current_trainloader.sampler.set_epoch(epoch)



#     train(config, epoch, config.TRAIN.END_EPOCH, 
#                 epoch_iters, base_lr, num_iters,
#                 trainloader, optimizer, model, writer_dict)

#     if flag_rm == 1 or (epoch % 2 == 0 and epoch <= 50)  or (epoch % 20 == 0 and epoch > 50 and epoch <= 180)  or (epoch>180 and epoch % 2 == 0) and (epoch>235): 
#         valid_loss, mean_IoU, IoU_array = validate(config, 
#                     testloader, model, writer_dict)

#     if flag_rm == 1:
#         flag_rm = 0
#     if args.local_rank <= 0:
#         logger.info('=> saving checkpoint to {}'.format(
#             final_output_dir + 'checkpoint_dhs_base_bdd.pth.tar'))
#         torch.save({
#             'epoch': epoch+1,
#             'best_mIoU': best_mIoU,
#             'state_dict': model.module.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }, os.path.join(final_output_dir,'checkpoint_dhs_base_bdd.pth.tar'))
#         if mean_IoU > best_mIoU:
#             best_mIoU = mean_IoU
#             torch.save(model.module.state_dict(),
#                     os.path.join(final_output_dir, 'best_dhs_base_bdd.pth'))
#             torch.save(model.module.state_dict(),
#                     os.path.join(final_output_dir, 'best_dhs_base_bdd.pt'))
#             if best_mIoU > 0.620:
#                 torch.save(model.module.state_dict(),
#                     os.path.join(final_output_dir, 'best_dhs_base_bdd_{: .6f}.pth'.format(best_mIoU)))

#                 torch.save({
#                 'epoch': epoch+1,
#                 'best_mIoU': best_mIoU,
#                 'state_dict': model.module.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 }, os.path.join(final_output_dir,'best_dhs_base_bdd_{: .6f}.pth.tar'.format(best_mIoU)))
#             else :
#                 torch.save({
#                 'epoch': epoch+1,
#                 'best_mIoU': best_mIoU,
#                 'state_dict': model.module.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 }, os.path.join(final_output_dir,'best_dhs_base_bdd.pth.tar'))
#         msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
#                     valid_loss, mean_IoU, best_mIoU)
#         logging.info(msg)




# if args.local_rank <= 0:

#     torch.save(model.module.state_dict(),
#             os.path.join(final_output_dir, 'final_dhs_base_bdd.pt'))

#     writer_dict['writer'].close()
#     end = timeit.default_timer()
#     logger.info('Hours: %d' % int((end-start)/3600))
#     logger.info('Done')

# if __name__ == '__main__':
# main()


# tic = time.time()
#     cur_iters = epoch*epoch_iters
#     writer = writer_dict['writer']
#     global_steps = writer_dict['train_global_steps']
#     record = []
#     for i_iter, batch in enumerate(trainloader, 0):
#         images, labels, _, _ = batch
#         images = images.cuda()
#         labels = labels.long().cuda()
        

#         losses, outputs, acc, loss_list = model(images, labels)
#         loss = losses.mean()
#         acc  = acc.mean()

#         if dist.is_distributed():
#             reduced_loss = reduce_tensor(loss)
#         else:
#             reduced_loss = loss

#         model.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # scheduler.step()
#         # measure elapsed time
#         batch_time.update(time.time() - tic)
#         tic = time.time()

#         # update average loss
#         ave_loss.update(reduced_loss.item())
#         ave_acc.update(acc.item())

#         avg_sem_loss1.update(loss_list[0].mean().item())   # 最终图像的loss
#         avg_sem_loss2.update(loss_list[1].mean().item())   # 第一个监督信号
#         avg_sem_loss3.update(loss_list[2].mean().item())   # 第二个监督信号
#         boundary_loss.update(loss_list[3].mean().item())   # 第三个监督信号

#         lr = adjust_learning_rate(optimizer,
#                                   base_lr,
#                                   num_iters,
#                                   i_iter+cur_iters)
#         if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
#             msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
#                   'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, loss1: {:.6f}, loss2: {:.6f}, SB loss: {:.6f}' .format(
#                       epoch, num_epoch, i_iter, epoch_iters,
#                       batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
#                       ave_acc.average(), avg_sem_loss1.average(), avg_sem_loss2.average(), avg_sem_loss3.average(), boundary_loss.average())
#             logging.info(msg)

#         record.append(loss)



    
#     writer.add_scalar('train_loss', ave_loss.average(), global_steps)
#     writer_dict['train_global_steps'] = global_steps + 1

# def validate(config, testloader, model, writer_dict):
#     model.eval()
#     ave_loss = AverageMeter()
#     pixel_acc = 0
#     nums = config.MODEL.NUM_OUTPUTS
#     confusion_matrix = np.zeros(
#         (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
#     confusion_matrix2 = np.zeros(
#         (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
#     mean_loss = 0
#     with torch.no_grad():
#         for idx, batch in enumerate(testloader):
#             image, label, _, _ = batch
#             size = label.size()
#             image = image.cuda()
#             label = label.long().cuda()
#             # bd_gts = bd_gts.float().cuda()

#             losses, pred, _, _ = model(image, label)
#             # pred[] = pred[1]
#             if not isinstance(pred, (list, tuple)):
#                 pred = [pred]   # 直接跳过

#             confusion_matrix += get_confusion_matrix(
#                 label,
#                 pred[1],
#                 size,
#                 config.DATASET.NUM_CLASSES,
#                 config.TRAIN.IGNORE_LABEL
#             )



#             loss = losses.mean()
#             if dist.is_distributed():
#                 reduced_loss = reduce_tensor(loss)
#             else:
#                 reduced_loss = loss
#             ave_loss.update(reduced_loss.item())


#     if dist.is_distributed():
#         confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
#         reduced_confusion_matrix = reduce_tensor(confusion_matrix)
#         confusion_matrix = reduced_confusion_matrix.cpu().numpy()
        


#     pos = confusion_matrix.sum(1)
#     res = confusion_matrix.sum(0)
#     tp = np.diag(confusion_matrix)
#     pixel_acc = tp.sum()/pos.sum()
#     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#     mean_IoU = IoU_array.mean()

#     if dist.get_rank() <= 0:
#         logging.info('acc:{} '.format(pixel_acc))
#         logging.info('{} {}'.format(IoU_array, mean_IoU))

#     writer = writer_dict['writer']
#     global_steps = writer_dict['valid_global_steps']
#     writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
#     writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
#     writer.add_scalar('acc', pixel_acc, global_steps)
#     writer_dict['valid_global_steps'] = global_steps + 1
#     return ave_loss.average(), mean_IoU, IoU_array
