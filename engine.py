import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from rich.progress import Progress
import utils
import time

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,num_classes=2):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    optimizer.zero_grad()
    start_time = time.time()
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes
    with Progress() as progress:
        task = progress.add_task(f"[green]Epoch {epoch} ", total=len(data_loader))

        for data_iter_step, (samples, targets) in enumerate(data_loader):
            progress.update(task, advance=1)
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step
            if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(samples)
                    loss = criterion(output, targets)
            else: # full precision
                output = model(samples)
                loss = criterion(output, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value): # this could trigger if using AMP
                print("Loss is {}, stopping training".format(loss_value))
                optimizer.zero_grad()  # Reset gradients
                continue  # Skip this batch
                # assert math.isfinite(loss_value)

            if use_amp:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            else: # full precision
                loss /= update_freq
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)

            torch.cuda.synchronize()


            _, preds = torch.max(output, 1)
            for i in range(num_classes):
                true_positives[i] += torch.sum((preds == i) & (targets == i)).item()
                false_positives[i] += torch.sum((preds == i) & (targets != i)).item()
                false_negatives[i] += torch.sum((preds != i) & (targets == i)).item()

            if mixup_fn is None:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
            else:
                class_acc = None
            metric_logger.update(loss=loss_value)
            metric_logger.update(class_acc=class_acc)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            if use_amp:
                metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(class_acc=class_acc, head="loss")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                if use_amp:
                    log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

            if wandb_logger:
                wandb_logger._wandb.log({
                    'Rank-0 Batch Wise/train_loss': loss_value,
                    'Rank-0 Batch Wise/train_max_lr': max_lr,
                    'Rank-0 Batch Wise/train_min_lr': min_lr
                }, commit=False)
                if class_acc:
                    wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
                if use_amp:
                    wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
                wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
    end_time = time.time() 
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats:{metric_logger},Time:{end_time - start_time}")

    # Calculate and print precision and recall for each class
    for i in range(num_classes):
        precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0
        recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0
        print(f'Class {i}: Precision: {precision:.5f}, Recall: {recall:.5f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, num_classes, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    # 初始化用于存储每类的真正例、假正例和假反例的计数器
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # 初始化用于存储平均精确率和召回率的 Meter 对象
    metric_logger.add_meter('avg_precision', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('avg_recall', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 切换到评估模式
    model.eval()
    # for data_iter_step, (images, target) in enumerate(data_loader):
    for batch in metric_logger.log_every(data_loader, 0, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 计算输出
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        # 转换输出为预测类别
        _, preds = torch.max(output, 1)

        # 更新每类的真正例、假正例和假反例计数
        for i in range(num_classes):
            true_positives[i] += torch.sum((preds == i) & (target == i)).item()
            false_positives[i] += torch.sum((preds == i) & (target != i)).item()
            false_negatives[i] += torch.sum((preds != i) & (target == i)).item()

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    # 计算并打印每类的精确率和召回率
    for i in range(num_classes):
        precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0
        recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0
        print(f'Class {i}: Precision: {precision:.5f}, Recall: {recall:.5f}')
        metric_logger.meters['avg_precision'].update(precision)
        metric_logger.meters['avg_recall'].update(recall)

    avg_precision = metric_logger.meters['avg_precision'].global_avg
    avg_recall = metric_logger.meters['avg_recall'].global_avg
    print(f'Average Precision: {avg_precision:.5f}, Average Recall: {avg_recall:.5f}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

