from cmath import exp
from torch.optim import lr_scheduler
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
         # 自定义函数
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponential_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
         # 自定义函数
        if current_step < num_warmup_steps:
            return math.exp(float(current_step)/float(max(1, num_warmup_steps) - 1)) 
        return max(0,0, math.exp((float(num_training_steps) - float(current_step)) / (float(num_training_steps) - float(num_warmup_steps)))/ math.e) 
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(args, optimizer):
    if args.lr_scheduler == 'lambda1':
        lambda1 = lambda epoch:np.sin(epoch) / (epoch + 1e-4)
        scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
    elif args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)
    elif args.lr_scheduler == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, 20, num_training_steps=args.num_epochs)
    return scheduler
    
