import time
import numpy as np
import torch
from torch.nn import functional as F
from train_utils import AverageMeter, save_checkpoint, inference, post_trans, cal_metrics
from monai.data import decollate_batch

def train_epoch(epoch, model, loader, optimizer, scheduler, loss_func, args):
    # Setup
    train_total_loss = AverageMeter('Loss', ':.4e')
    train_mse_loss = AverageMeter('MSELoss', ':.4e')
    model.train()
    start_time = time.perf_counter()
    for idx, batch in enumerate(loader):
        inputs, labels = batch['image'].float().cuda(), batch['label'].float().cuda()
        optimizer.zero_grad()
        mse_loss = 0.0
        if args.use_cl:
            num_channel = inputs.shape[1]
            preds_r, deep_feature_r = model(inputs[:, :num_channel//2, ...])
            preds_p, deep_feature_p = model(inputs[:, num_channel//2:, ...])
            preds = []
            for p in range(len(preds_r)):
                preds.append(torch.cat([preds_r[p], preds_p[p]], dim=0))
            labels = torch.cat([labels, labels], dim=0)
            mse_loss = F.mse_loss(deep_feature_r, deep_feature_p, reduction = "none").mean()
        else:
            preds, _ = model(inputs)

        loss = 0
        for p in range(len(preds)):
            loss += loss_func(preds[p], labels, is_train=True)
        loss += mse_loss * 0.1

        loss.backward()  
        optimizer.step()

        train_total_loss.update(loss.item())
        train_mse_loss.update(mse_loss)
        print(
                "Train epoch [{}/{}]({}/{}): ".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(train_total_loss.avg),
                "time {:.2f}s".format(time.perf_counter() - start_time),
            )
        start_time = time.perf_counter()
    if scheduler is not None:
        scheduler.step()
    return train_total_loss.avg


def val_epoch(data_loader, model, loss_func, metric, epoch, args, save_folder=None, patch_shape = 96):
    # Setup
    losses = AverageMeter('Loss', ':.4e')
    
    metrics = []
    
    start_time = time.perf_counter()
    for idx, val_data in enumerate(data_loader):

        model.eval()
        with torch.no_grad():
            val_inputs = val_data["image"].cuda()
            val_labels = val_data["label"].cuda()
            val_outputs = inference(val_inputs, model, patch_shape = patch_shape)
            val_preds = [post_trans(i) for i in decollate_batch(val_outputs)]

            loss_ = loss_func(val_outputs, val_labels, is_train=False)

        losses.update(loss_.item())
        metric_ = metric(val_outputs, val_labels)
        metrics.extend(metric_)

        print(
            "Valid epoch [{}/{}]({}/{}): ".format(epoch, args.max_epochs, idx, len(data_loader)),
            "loss: {:.4f}".format(losses.avg),
            "time {:.2f}s".format(time.perf_counter() - start_time),
        )
        start_time = time.perf_counter()

    class_avg_metrics = cal_metrics(metrics, epoch, save_folder)


    return losses.avg, np.mean(class_avg_metrics)


def run_training(
        model,
        start_epoch,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        eval_metric,
        args
    ):
    print("Start training ...")

    best_val_metric = 0.0
    for epoch in range(start_epoch + 1, args.max_epochs+1):
        train_epoch(epoch, model, train_loader, optimizer, scheduler, loss_func, args)

        # Validate at the end of epoch every eval interval
        if epoch % args.eval_interval == 0:

            val_loss, val_dice = val_epoch(val_loader, model, 
                    loss_func, eval_metric, epoch, args,
                    save_folder=args.ckpt_save_dir, 
                    patch_shape = args.patch_shape)

            if val_dice > best_val_metric:
                print(f"Saving {epoch} epoch with best metrics {val_dice}")
                best_val_metric = val_dice
                save_checkpoint(
                    dict(
                        epoch=epoch,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                    ), 
                    save_folder=args.ckpt_save_dir)
                
    print(f'Training Finished ! Best Val Dice: {best_val_metric}')

    return best_val_metric

