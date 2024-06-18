import sys
import math
from sklearn.metrics import precision_score, recall_score,classification_report
import torch
from tqdm import tqdm
from torch.optim import Optimizer
import numpy as np
import random
import torch.distributed as dist



class Lion(Optimizer):
  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)

        p.add_(torch.sign(update), alpha=-group['lr'])
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss

class Lamb(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instead."  # noqa: E501
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = group["lr"]
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])
                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        # self.class_weight = torch.tensor([1.0, 1.0])

    def forward(self, x, target):
        one_hot = torch.zeros_like(x).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (
            x.shape[1] - 1
        )
        # device = x.device
        # class_weight = self.class_weight.to(device)nn.BCEWithLogitsLoss() 
        # return torch.nn.functional.binary_cross_entropy_with_logits(x, smooth_target, weight=class_weight, reduction='mean')  # noqa: E501
        return torch.nn.functional.binary_cross_entropy_with_logits(
            x, smooth_target, reduction="mean"
        )


class RASampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 重复采样后每个replica的样本量
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        # 重复采样后的总样本量
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # 每个replica实际样本量，即不重复采样时的每个replica的样本量
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]  # 重复3次
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample: 使得同一个样本的重复版本进入不同的进程（GPU）
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])  # 截取实际样本量

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def mixup_data(x, y, alpha=0.1, device="cuda"):
    # 随机生成一个 beta 分布的参数 lam，用于生成随机的线性组合，以实现 mixup 数据扩充。
    lam = np.random.beta(alpha, alpha)
    # 生成一个随机的序列，用于将输入数据进行 shuffle。
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    # 得到混合后的新图片
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # 得到混图对应的两类标签
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(data, target, alpha=1.0):
    def cutmix_batch(data, target):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_target = target[indices]
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        return data, target, shuffled_target, lam

    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    return cutmix_batch(data, target)


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    lr_scheduler,
    loss_fn,
    mixup=False,
    cutmix=False,
):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    scaler = torch.cuda.amp.GradScaler()
    y_true = []
    y_pred = []

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if mixup and cutmix:
                # 随机选择使用mixup还是cutmix
                if random.random() < 0.5:
                    # 使用mixup
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, alpha=0.1, device=device
                    )
                    loss = mixup_criterion(loss_fn, model(images), labels_a, labels_b, lam)
                    pred = model(images)
                else:
                    # 使用cutmix
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
                    loss = cutmix_criterion(loss_fn, model(images), labels_a, labels_b, lam)
                    pred = model(images)
            elif mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=0.1, device=device
                )
                loss = mixup_criterion(loss_fn, model(images), labels_a, labels_b, lam)
                pred = model(images)

            elif cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
                loss = cutmix_criterion(loss_fn, model(images), labels_a, labels_b, lam)
                pred = model(images)

            else:
                # 未使用mixup和cutmix
                pred = model(images)
                loss = loss_fn(pred, labels)

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        y_true.extend(labels.tolist())
        y_pred.extend(pred_classes.tolist())
        # 不使用混合精度训练
        # loss.backward()
        # optimizer.step()
        # 使用混合精度训练
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        accu_loss += loss.detach()
        data_loader.desc = (
            "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"],
            )
        )
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        lr_scheduler.step()

    # 计算精确率和召回率F1-score 
    report = classification_report(y_true, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            print(f"Class {label} - Precision: {precision:.3f}, Recall: {recall:.3f}")

    return (
        accu_loss.item() / (step + 1),
        accu_num.item() / sample_num,
        lr_scheduler.get_last_lr()[0],
    )


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    y_true = []
    y_pred = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        y_true.extend(labels.tolist())
        y_pred.extend(pred_classes.tolist())
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
    # 计算所有类别的精确率和召回率
    report = classification_report(y_true, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            print(f"类别 {label} - 精确率: {precision:.3f}, 召回率: {recall:.3f}")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
    end_factor=1e-6,
):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = x - warmup_epochs * num_step
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (
                1 - end_factor
            ) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }
    # 记录对应的权重名称
    parameter_group_names = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
