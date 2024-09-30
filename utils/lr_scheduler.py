import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineWithLinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        super(CosineWithLinearWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        if current_step <= self.warmup_steps:
            # Linear warmup phase
            lr = self.max_lr * current_step / self.warmup_steps
        else:
            # Cosine decay phase
            decay_steps = current_step - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_steps / total_decay_steps))
            lr = self.max_lr * cosine_decay
        return [lr for _ in self.base_lrs]
