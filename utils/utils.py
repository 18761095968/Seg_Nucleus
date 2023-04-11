import torch
import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    # 在训练模型的过程中，可能会发生梯度爆炸的情况，导致模型训练失败
    # 梯度截断Clip, 将梯度约束在某一个区间之内，在训练的过程中，在优化器更新之前进行梯度截断操作。
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                # 其中的 clamp_ 操作就可以将梯度约束在[ -grad_clip, grad_clip] 的区间之内。大于grad_clip的梯度，将被修改等于grad_clip。
                param.grad.data.clamp_(-grad_clip, grad_clip)

# 调整学习率
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

# 采用自定义的AvgMeter类来管理一些变量的更新
# 读取某个变量的时候，通过对象属性的方式来读取
class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        # 初始化的时候调用reset
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

# 计算给定模型和输入张量的总参数数量。
def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))